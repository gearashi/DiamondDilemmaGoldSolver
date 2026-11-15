"""Prove native dispatch pages cover exactly the writable lanes and can stop safely."""
import os
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from dfs_gpu import dispatch_pages
from test_r35_checkpoint_ranges import NativeControls
from test_gpu_reasoning import varied_gold_fixture


class PagePlanTests(unittest.TestCase):
    def test_full_width_sparse_dense_and_pending_pages(self):
        generator=np.random.default_rng(319)
        for size in (1,127,128,129,1024,1025,32768,131072):
            for budget in (1,8,63,64,511,512):
                for flags in (0,12,31):
                    states=generator.integers(0,4,size=size,dtype=np.uint8)
                    if size>128:
                        states[:size//2]=3
                        states[size//3]=1
                    gpu=SimpleNamespace(n=size,feature_bits=flags,states=SimpleNamespace(get=lambda: states))
                    cover=np.zeros(size,np.int8)
                    for start,length in dispatch_pages(gpu,budget):
                        self.assertEqual(start%128,0)
                        self.assertTrue(0<length<=size-start)
                        cover[start:start+length]+=1
                        if flags:
                            self.assertLessEqual(length,1024)
                            self.assertLessEqual(length*budget,max(32768,128*budget))
                    self.assertTrue(np.all(cover[states<=1]==1))
                    self.assertLessEqual(cover.max(),1)


class NativePageChecks:
    pool=NativeControls.pool

    def test_subdividing_and_stopping_preserve_every_physical_field(self):
        data,board=varied_gold_fixture()
        n=512
        ids=np.array([4,5,128,129,259,384,489],np.int32)
        gpu=self.pool(data,board,n,gold_data=data,gpu_features={'dynamic_mrv':True,'forced':True})
        gpu.load_prefixes([[3*i] for i in range(len(ids))],lanes=ids)
        source=gpu.snapshot_checkpoint()
        module='dfs_webgpu' if self.web else 'dfs_gpu'
        full=[(0,n)]
        pages=[(i,128) for i in range(0,n,128)]
        with patch(module+'.dispatch_pages',return_value=full):
            gpu.step(24)
        direct=gpu.snapshot_checkpoint()
        gpu.restore_checkpoint(source)
        with patch(module+'.dispatch_pages',return_value=pages):
            gpu.step(24)
        whole=gpu.snapshot_checkpoint()
        for field in gpu.DEVICE_FIELDS:
            np.testing.assert_array_equal(whole[field],direct[field],err_msg=field)
        gpu.restore_checkpoint(source)
        gpu._page_stop=lambda:True
        with patch(module+'.dispatch_pages',return_value=pages):
            gpu.step(24)
        partial=gpu.snapshot_checkpoint()
        for field in gpu.DEVICE_FIELDS:
            np.testing.assert_array_equal(partial[field][...,:128],direct[field][...,:128],err_msg=field)
            np.testing.assert_array_equal(partial[field][...,128:],source[field][...,128:],err_msg=field)
        resumed=self.pool(data,board,n,gold_data=data,gpu_features={'dynamic_mrv':True,'forced':True})
        resumed.restore_checkpoint(partial)
        with patch(module+'.dispatch_pages',return_value=pages[1:]):
            resumed.step(24)
        restored=resumed.snapshot_checkpoint()
        for field in gpu.DEVICE_FIELDS:
            np.testing.assert_array_equal(restored[field],direct[field],err_msg=field)


@unittest.skipUnless(os.environ.get('DIAMOND_TEST_CUDA')=='1','requires CUDA')
class CudaPageTests(NativePageChecks,unittest.TestCase):
    web=False

@unittest.skipUnless(os.environ.get('DIAMOND_TEST_WEBGPU')=='1','requires WebGPU')
class WebPageTests(NativePageChecks,unittest.TestCase):
    web=True
