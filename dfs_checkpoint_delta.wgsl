@group(0) @binding(0) var<storage,read> source:array<u32>;
@group(0) @binding(1) var<storage,read_write> previous:array<u32>;
@group(0) @binding(2) var<storage,read> plan:array<u32>;
struct Output {failed:atomic<u32>, data:array<u32>,}
@group(0) @binding(4) var<storage,read_write> scratch:array<u32>;
@group(0) @binding(3) var<storage,read_write> result:Output;

fn price(v:u32)->vec2<u32>{
    let change=v&255u;let pop=(v>>8u)&255u;let used=(v>>16u)&31u;
    if change==0u && used==0u {return vec2<u32>(0u);}
    let bits=select(9u,16u,(v>>21u)!=0u);
    var count=change;var mode=0u;var best=((bits+8u)*change+31u)/32u;
    for(var m=1u;m<5u;m++){
        let c=select(select(change,pop,(m&1u)!=0u),160u,m==4u);
        let cost=((bits+select(0u,8u,m<2u))*c+31u)/32u+select(0u,5u,m==2u || m==3u);
        if cost<best {best=cost;mode=m;count=c;}
    }
    let ticket=1u+used+32u*(10u*count+mode+select(0u,5u,bits==16u));
    return vec2<u32>(ticket,best+countOneBits(used));
}
var<workgroup> prefix_words:array<u32,128>;
var<workgroup> prefix_dirty:array<u32,128>;
fn lane_at(i:u32)->u32 {
    if (plan[2]&0x80000000u)!=0u {return (plan[2]&0x7fffffffu)+i;}
    return plan[3u+i];
}
@compute @workgroup_size(128)
fn price_groups(@builtin(local_invocation_index) j:u32,@builtin(workgroup_id) group:vec3<u32>) {
    let i=group.x*128u+j;let k=plan[1];var p=vec2<u32>(0u);
    if i<k {
        let lane=lane_at(i);let n=plan[0];
        var changes:array<u32,5>;var occupied:array<u32,5>;
        var change=0u;var pop=0u;var bad=0u;var used=0u;
        for(var g=0u;g<5u;g++) {
            var diff=0u;var live=0u;
            for(var j=0u;j<32u;j++) {
                let at=(32u*g+j)*n+lane;let now=source[at]&65535u;
                if now!=(previous[at]&65535u) {diff|=1u<<j;previous[at]=now;}
                if now!=65535u {live|=1u<<j;bad|=u32(now>=480u);}
            }
            changes[g]=diff;occupied[g]=live;change+=countOneBits(diff);pop+=countOneBits(live);
            let at=(160u+g)*n+lane;
            if source[at]!=previous[at] {used|=1u<<g;previous[at]=source[at];}
        }
        p=price(change|(pop<<8u)|(used<<16u)|(bad<<21u));
        if p.x!=0u {
            let mode=(((p.x-1u)>>5u)%10u)%5u;
            if mode!=4u {
                for(var g=0u;g<5u;g++){scratch[(4u+g)*k+i]=select(changes[g],occupied[g],(mode&1u)!=0u);}
            }
        }
    }
    prefix_words[j]=p.y;prefix_dirty[j]=u32(p.x!=0u);workgroupBarrier();
    for(var d=1u;d<128u;d*=2u){
        var w=0u;var t=0u;
        if j>=d {w=prefix_words[j-d];t=prefix_dirty[j-d];}
        workgroupBarrier();prefix_words[j]+=w;prefix_dirty[j]+=t;workgroupBarrier();
    }
    if i<k {
        scratch[4u*i]=p.x;scratch[4u*i+1u]=prefix_words[j]-p.y;
        scratch[4u*i+2u]=prefix_dirty[j]-u32(p.x!=0u);scratch[4u*i+3u]=p.y;
    }
    if j==127u {
        scratch[9u*k+2u*group.x]=prefix_words[j];scratch[9u*k+2u*group.x+1u]=prefix_dirty[j];
    }
}
@compute @workgroup_size(1)
fn scan_groups(){
    let k=plan[1];var words=0u;var dirty=0u;
    for(var g=0u;g<(k+127u)/128u;g++){
        let at=9u*k+2u*g;let w=scratch[at];let d=scratch[at+1u];
        scratch[at]=words;scratch[at+1u]=dirty;words+=w;dirty+=d;
    }
    let sparse=select(1u,2u,k>65536u);let dense=(k+1u)/2u;
    let form=select(sparse,0u,dense<=sparse*dirty);
    atomicStore(&result.failed,0u);
    result.data[0]=dirty;result.data[1]=words;result.data[2]=select(dense,form*dirty,form!=0u);result.data[3]=form;
}
@compute @workgroup_size(64)
fn pack_groups(@builtin(global_invocation_id) id:vec3<u32>){
    let i=id.x;let k=plan[1];
    if i>=k {return;}
    let ticket=scratch[4u*i];let form=result.data[3];
    if form==0u && (i&1u)==0u {
        var v=ticket;
        if i+1u<k {v|=scratch[4u*(i+1u)]<<16u;}
        result.data[4u+i/2u]=v;
    }
    if ticket==0u {return;}
    let at=9u*k+2u*(i/128u);let before=scratch[at]+scratch[4u*i+1u];
    let idx=scratch[at+1u]+scratch[4u*i+2u];
    if form==1u {result.data[4u+idx]=(i<<16u)|ticket;}
    if form==2u {result.data[4u+2u*idx]=i;result.data[5u+2u*idx]=ticket;}
    let v=ticket-1u;let choice=(v>>5u)%10u;
    let flags=choice%5u+select(0u,8u,choice>=5u)+((v&31u)<<4u)+((v/320u)<<10u);
    let base=4u+result.data[2]+before;
    let mode=flags&7u;let bits=select(9u,16u,(flags&8u)!=0u);let used=(flags>>4u)&31u;
    let n=plan[0];let lane=lane_at(i);let limit=base+scratch[4u*i+3u];
    var pos=base+select(0u,5u,mode==2u || mode==3u);var low=0u;var filled=0u;
    for(var g=0u;g<5u;g++) {
        var mask=0xffffffffu;
        if mode!=4u {mask=scratch[(4u+g)*k+i];}
        if mode==2u || mode==3u {result.data[base+g]=mask;}
        while mask!=0u {
            let row=g*32u+firstTrailingBit(mask);mask&=mask-1u;
            let now=source[row*n+lane]&65535u;let code=select(now,480u,bits==9u && now==65535u);
            let token=select(code,(row<<bits)|code,mode<2u);let width=bits+select(0u,8u,mode<2u);
            low|=token<<filled;
            if filled+width>=32u {
                if pos>=limit {atomicOr(&result.failed,1u);return;}
                result.data[pos]=low;pos++;low=select(token>>(32u-filled),0u,filled+width==32u);
            }
            filled=(filled+width)&31u;
        }
    }
    if filled!=0u {if pos>=limit {atomicOr(&result.failed,1u);return;}result.data[pos]=low;pos++;}
    for(var g=0u;g<5u;g++) {if (used&(1u<<g))!=0u {
        if pos>=limit {atomicOr(&result.failed,1u);return;}result.data[pos]=source[(160u+g)*n+lane];pos++;
    }}
    if pos!=limit {atomicOr(&result.failed,1u);}

}
