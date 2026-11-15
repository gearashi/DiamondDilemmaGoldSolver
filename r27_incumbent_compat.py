"""Checked source compatibility for local v1.7 heuristic continuations."""
from functools import lru_cache
from hashlib import sha256
from pathlib import Path

_COMPATIBLE = {
    'constraint_lns_resume_round25.py': ('4089aea03f01439e55e2b99f96ce37abbf19270566b14809feedddfe56b85549', 'e18dcf659c75ccf746f5d0bbd3f6acd1263adee7a60a414e4fb8dc2d3f7007e4'),
    'constraint_repair_gold_round26.py': ('d22fcc9530e87c142d28579d4d606ecccf1708318903b66f469133bc24984f09', '7d99679e8aa9c860a02afc4e1c37b90774c622882d01ca3cf7358da83490da02'),
    'constraint_repair_resume_round25.py': ('bf44954f046b2dbc0140d2b426f8c58634ba1d232693887691b1fc02c993ba85', '6fbd361b87b861de2ccd31036637d5e66c5780e011cc3b5e61fcb8c75fb25b99'),
    'constraint_repair_service_round26.py': ('e89b54451737174811f0e2a5bd263e16079e8b23b7727518127299e26bd817c4', '8b31c913c7445fe388f85a49d63ddacc1b06c177319faf180455600b2ff69cdf'),
    'gpu_hybrid_resume_round25.py': ('2113de18bd8aeba663f0d693de842faf6c3600e37f8ed55b92d7e88d655727c0', '1f3355aa34d0dd0f4df9713dbbaabca8d887a004deb5cb85a7bb364ddef0f525'),
    'gpu_sampling.py': ('6a1a0b5763f995fdfb681958c504d21750eb9dab6073524e3051516efa3b5e68', '4f8e26b1290f3ac4c5df49c959f83d75d84dd3de606193586b5201315e8c21b0'),
    'gpu_sampling_resume_round25.py': ('f154a702fac63d2b8700815d9807c858bbaf5f817e0d234fd84ecc57a7bb7b98', '119c050605e99346ad0e9fe7a99d76753c09e4e04409409e81dec1acdc75da1c'),
    'gpu_sampling_state_round25.py': ('145dffcfbe630d77c748bbaecc91d458d243e21b18151ce3f5c1a2f0b39564c1', '673e63399e323d80b630e91e5f393961033f5fe39f3ebc7fa6c9526ad194183b'),
    'sampling_crossover_round25.py': ('ddbb92d6da115f3d93a2694a153bbd84a3ab891f5bedf20437503cd6d7ae92b4', '37beec945027144fb62a5c3be513967d5fad9d182bec3e4ad07edd94b277416c'),
    'sampling_crossover_round26.py': ('4895892f124c9f6547dcaddddea76b500b2ef6f74a5e86e0145798bbcec8242f', 'a4fe4bff5638005fef1f2d3af911f718eec4d0c569a711f3a6d3fb7d5848bb96'),
    'search_options.py': ('18455f7a07bf56e1e2364d24f67ac1dfd222a3420a6bea8003dbdaa8ac81eaca', 'a031ac39552a9f028191a4b62ff29a096777c7e3bbc671c0fd0ced4496b00e81'),
}
_V17_CRLF = {
    'constraint_lns_resume_round25.py': 'bb1d187b26aa1719aee81d479a79ba2cf238252a022816a3fc4c0daeefb64267',
    'constraint_repair_gold_round26.py': '35f49081cc151c69d4cacff5d03175996b5856aca5adbf20875f5b526abbd9f0',
    'constraint_repair_resume_round25.py': 'd401492a327afff88b8f0894e0e42dd765a97926f4eabd977995a63428b602e2',
    'constraint_repair_service_round26.py': '380fe2e8f4789c9a2b4be4f86a9e931984e1356eb89c15f5074e5a42d0183604',
    'gpu_hybrid_resume_round25.py': 'f82ac66caf9d6959c7ca9cddfcdcba5e830c12f239f0a96afd102d955093da5f',
    'gpu_sampling.py': 'd55db6fccf05ea980c52f70896db69010a360e98b3251900dc62ff8a4ead24a6',
    'gpu_sampling_resume_round25.py': 'f174bc5ca39f20896fbd463bdbca22539b6006490242b9fd09f53aff236a1cb8',
    'gpu_sampling_state_round25.py': 'eba65d5847195873558fd4e9632084574dfc368e19748dcd9e4ae5058c0e8691',
    'sampling_crossover_round25.py': 'fb8049cb605b065049db5ff57f8ab052fc9870d83e6e2e8fcdfbc32fd48cfee4',
    'sampling_crossover_round26.py': 'b365dda6955078b4b272708bfef25e534ba2e63150de5747d187fbbb483c3b83',
    'search_options.py': '7eef5cb6fd64667eb75f42abfa65c7aaa80a6a2b59cb3cc18774451d494791ac',
}
_RELEASE_TEXT = {
    'constraint_advanced.py': ('2667a466808388ef85300d347a84ab21ee2c8c4eed0d402876b04edce90b4e29', '2667a466808388ef85300d347a84ab21ee2c8c4eed0d402876b04edce90b4e29', '2667a466808388ef85300d347a84ab21ee2c8c4eed0d402876b04edce90b4e29'),
    'constraint_circuit.py': ('acce965e47024e6f16737d522145f68a15867a9eaa37f15888cee53601a49aa8', 'acce965e47024e6f16737d522145f68a15867a9eaa37f15888cee53601a49aa8', 'acce965e47024e6f16737d522145f68a15867a9eaa37f15888cee53601a49aa8'),
    'constraint_connectivity_round26.py': ('d6a88b0180324d663ca1039b2e17ce443c1a6058738042c21d83b841c9c6a5d3', 'd6a88b0180324d663ca1039b2e17ce443c1a6058738042c21d83b841c9c6a5d3', '89f363f0419b30a84ab42a80d7cb0928abf7c94fcbc7480055671fd6909cfdf7'),
    'constraint_deep.py': ('14797886c714051e31226bca84a4c86f4672cdc83e3f7caac26d600db62fe5a1', '14797886c714051e31226bca84a4c86f4672cdc83e3f7caac26d600db62fe5a1', '14797886c714051e31226bca84a4c86f4672cdc83e3f7caac26d600db62fe5a1'),
    'constraint_explanations.py': ('1d1001a41eb6dcb1be27c68d926f3c6468d7b8d9281961eef7a89584f42d5b95', '1d1001a41eb6dcb1be27c68d926f3c6468d7b8d9281961eef7a89584f42d5b95', '1d1001a41eb6dcb1be27c68d926f3c6468d7b8d9281961eef7a89584f42d5b95'),
    'constraint_gold_graph.py': ('f55629f711c2f87738818a6ceecdaf650f48be742ef5cee616db5d147dbbbf70', 'f55629f711c2f87738818a6ceecdaf650f48be742ef5cee616db5d147dbbbf70', 'f55629f711c2f87738818a6ceecdaf650f48be742ef5cee616db5d147dbbbf70'),
    'constraint_incremental_hall.py': ('8d09da9c76989bddfe6fa71e61c93f40bc45ce12c7f5d7260a9f8c420734a5f5', '8d09da9c76989bddfe6fa71e61c93f40bc45ce12c7f5d7260a9f8c420734a5f5', '8d09da9c76989bddfe6fa71e61c93f40bc45ce12c7f5d7260a9f8c420734a5f5'),
    'constraint_lns.py': ('e7d900252543b566fdb3e049552dbd223168a001e170712a8a10fe5210f383ed', 'e7d900252543b566fdb3e049552dbd223168a001e170712a8a10fe5210f383ed', '7a918f9146bffe1e8a6d119ed1fe4719c04d40f28d181ad696f8a72e71279e81'),
    'constraint_lns_resume_round25.py': ('e18dcf659c75ccf746f5d0bbd3f6acd1263adee7a60a414e4fb8dc2d3f7007e4', '4089aea03f01439e55e2b99f96ce37abbf19270566b14809feedddfe56b85549', 'bb1d187b26aa1719aee81d479a79ba2cf238252a022816a3fc4c0daeefb64267'),
    'constraint_native.c': ('76575f2913cd1544362d264e6d3443e5a32861f9718314f8f2277ea438d1aa28', '76575f2913cd1544362d264e6d3443e5a32861f9718314f8f2277ea438d1aa28', 'bc058d7d29f6ca6e1712067a37d5faa8d7cfd11137657fc32012702d839e2337'),
    'constraint_native.py': ('7d7d41842f918d625cb296aa8dff9ff91d90cf284715619a6265a2e0d684de6c', '7d7d41842f918d625cb296aa8dff9ff91d90cf284715619a6265a2e0d684de6c', '7d7d41842f918d625cb296aa8dff9ff91d90cf284715619a6265a2e0d684de6c'),
    'constraint_parity.py': ('e06b98324621f542a42d686892ed4b92b772d7214ba5ef90fe27a789588ed820', 'e06b98324621f542a42d686892ed4b92b772d7214ba5ef90fe27a789588ed820', 'e06b98324621f542a42d686892ed4b92b772d7214ba5ef90fe27a789588ed820'),
    'constraint_pruning.py': ('c64363073be2fca374ca688917fcb35785a44da326baea6c0bcb9e7d452d41e3', 'c64363073be2fca374ca688917fcb35785a44da326baea6c0bcb9e7d452d41e3', 'c64363073be2fca374ca688917fcb35785a44da326baea6c0bcb9e7d452d41e3'),
    'constraint_region_branch_price_round26.py': ('840470613b2d52047f74658a8af35df6199de2222d1782cec58b881ee6d33b5a', '840470613b2d52047f74658a8af35df6199de2222d1782cec58b881ee6d33b5a', '4b391d30d58b6eea28a6475c418a05a9281396e6910ad5b966ec1b43a9ec71dd'),
    'constraint_region_certificate.py': ('ef2277a6a846fa6d46dd39bf0351fbd20220c6a9bbb7d3f81d28c90b30fdbac1', 'ef2277a6a846fa6d46dd39bf0351fbd20220c6a9bbb7d3f81d28c90b30fdbac1', 'ef2277a6a846fa6d46dd39bf0351fbd20220c6a9bbb7d3f81d28c90b30fdbac1'),
    'constraint_region_compact.py': ('00ed09f8bd1ae2ed0a7af66d1bdaa488395bc4d10352c51359aa7a7801f59652', '00ed09f8bd1ae2ed0a7af66d1bdaa488395bc4d10352c51359aa7a7801f59652', '706b24d5870b57cd1d4f1809b42266de59e70584c473fd13c7225f66d4998d02'),
    'constraint_region_compiler_round26.py': ('43e330dae1937ba9462eb44060086cf0932efe3aba0d49c04a157fc107d95339', '43e330dae1937ba9462eb44060086cf0932efe3aba0d49c04a157fc107d95339', '2cdebdac281c406c666b17732bbded84768a721f1deed812ca1947ea26d197b4'),
    'constraint_region_cursor.py': ('63305556c643eea09bb445a6aa48388259bfaf251b0ce8c6535348f94a07d61e', '63305556c643eea09bb445a6aa48388259bfaf251b0ce8c6535348f94a07d61e', '63305556c643eea09bb445a6aa48388259bfaf251b0ce8c6535348f94a07d61e'),
    'constraint_region_cuts.py': ('c2b156837479dbf0368657c7d473b4fb028a1e3e2957d5220b6278b85d08ac74', 'c2b156837479dbf0368657c7d473b4fb028a1e3e2957d5220b6278b85d08ac74', 'c2b156837479dbf0368657c7d473b4fb028a1e3e2957d5220b6278b85d08ac74'),
    'constraint_region_incremental.py': ('fd864e36d996ce8c34b63f23be32f4fe49970beceae06a4f4483e9af29f97d8c', 'fd864e36d996ce8c34b63f23be32f4fe49970beceae06a4f4483e9af29f97d8c', 'fd864e36d996ce8c34b63f23be32f4fe49970beceae06a4f4483e9af29f97d8c'),
    'constraint_region_integer_round26.py': ('28672a3f3ce1b234b773979f4747d530c55cae6e3c2ff97a296700fe1df7793b', '28672a3f3ce1b234b773979f4747d530c55cae6e3c2ff97a296700fe1df7793b', '318b029b909b62f9d2d114b24b033f1a141375e725ed0c8235c7e60712728e5a'),
    'constraint_region_interfaces_round26.py': ('6f5c25db75d439c5b0ed9a3e863c3741026dafe0da19054bd3a35b5f82f1aff3', '6f5c25db75d439c5b0ed9a3e863c3741026dafe0da19054bd3a35b5f82f1aff3', '49c1c0e12c7aebc1860a92d9ac1cd12c870a4ae57d9d7941881baace784194cc'),
    'constraint_region_isomorphism.py': ('ac71efe5579ed7d6c95667f512187cfb5c25d8d7e82d2ac11a506ceb78eec16f', 'ac71efe5579ed7d6c95667f512187cfb5c25d8d7e82d2ac11a506ceb78eec16f', 'ac71efe5579ed7d6c95667f512187cfb5c25d8d7e82d2ac11a506ceb78eec16f'),
    'constraint_region_master.py': ('c5eef7f7b49480bc8e24cd6ab54810115b088c7017e9a32c198d9c9502c6be54', 'c5eef7f7b49480bc8e24cd6ab54810115b088c7017e9a32c198d9c9502c6be54', '94f5dd2524c1bff0ca51bd4dd383f9079d2a8a79126fc896ff27be62027ba732'),
    'constraint_region_messages_round26.py': ('99efe3b109d4b929eeb2368063f04949c89fbff8893d4a3640a20c8d1181251e', '99efe3b109d4b929eeb2368063f04949c89fbff8893d4a3640a20c8d1181251e', '2416501b44ac1268f8a13262367635e62d2b040c7af14b2ad1c7a70afa7d8fed'),
    'constraint_region_models.py': ('9136121f3ec9c7e476c1c519c7ec70b83a6b4833fd8085b59808d820b2c56a7f', '9136121f3ec9c7e476c1c519c7ec70b83a6b4833fd8085b59808d820b2c56a7f', '8c3dae3b84ddc00837510a94bcb3d3f5e8b0a7d4317af8bcfde53f637f8f8c58'),
    'constraint_region_multichain.py': ('a1c9fc08d16da4a4b734ca00e8bd7568adb4cd653d3de80baef3cbf001b61e32', 'a1c9fc08d16da4a4b734ca00e8bd7568adb4cd653d3de80baef3cbf001b61e32', 'a1c9fc08d16da4a4b734ca00e8bd7568adb4cd653d3de80baef3cbf001b61e32'),
    'constraint_region_network.py': ('73b08cd343cf39c02fb9d325479b5f9f6cec40d892c2d2b7f6d38ca2b06b15fa', '73b08cd343cf39c02fb9d325479b5f9f6cec40d892c2d2b7f6d38ca2b06b15fa', '73b08cd343cf39c02fb9d325479b5f9f6cec40d892c2d2b7f6d38ca2b06b15fa'),
    'constraint_region_oracles.py': ('9c74447933688fba0f654484ff841308dacd6d2e5af7749233b7de3d5f1b4cb4', '9c74447933688fba0f654484ff841308dacd6d2e5af7749233b7de3d5f1b4cb4', '9c74447933688fba0f654484ff841308dacd6d2e5af7749233b7de3d5f1b4cb4'),
    'constraint_region_proof_round26.py': ('45e680a7d42cedebbd2556d544120025b35ba33ed70d2049802437fc69f23a16', '45e680a7d42cedebbd2556d544120025b35ba33ed70d2049802437fc69f23a16', 'd507ce211a37cb7b20c34f7c9f8119ff38cdd97c2a16a062abf85d05447239d2'),
    'constraint_region_rank.py': ('8d25cfe66ccaa0dc94a505a25d06b5c3f78765a52d8703bc3f7401d4554aee5a', '8d25cfe66ccaa0dc94a505a25d06b5c3f78765a52d8703bc3f7401d4554aee5a', '8d25cfe66ccaa0dc94a505a25d06b5c3f78765a52d8703bc3f7401d4554aee5a'),
    'constraint_region_store.py': ('55d4e8183c8304426ed5642f90022a84fcf14b540e7cce2d8b104a3de4c35b37', '55d4e8183c8304426ed5642f90022a84fcf14b540e7cce2d8b104a3de4c35b37', '55d4e8183c8304426ed5642f90022a84fcf14b540e7cce2d8b104a3de4c35b37'),
    'constraint_region_symbolic.py': ('8c771d77a3043c45cb140ddefda6739069b70c28ffea18675a92a3cd7684b5ca', '8c771d77a3043c45cb140ddefda6739069b70c28ffea18675a92a3cd7684b5ca', '8c771d77a3043c45cb140ddefda6739069b70c28ffea18675a92a3cd7684b5ca'),
    'constraint_region_tree_round26.py': ('42320fef01fc6a1a06cac7bc7104229c5fcd1724a796fce33e7112722b6a5a9d', '42320fef01fc6a1a06cac7bc7104229c5fcd1724a796fce33e7112722b6a5a9d', '8064110f0c21175552b8f253c55083c3bd91e8fed9512a78fc31c3be5df272a5'),
    'constraint_regional_proof_bridge_round26.py': ('5560778894976a20f00af0d51a3b101da2ef675eeac70a950e42557d5627ab53', '5560778894976a20f00af0d51a3b101da2ef675eeac70a950e42557d5627ab53', 'c02b2a651c273d5a2a21c57b2ab1da7d4aa62a68662a6cbb2fb3072d865a7746'),
    'constraint_regions.py': ('8afda0e3a6cef2260c0036302a16ff4f3cb526f8726c3b007ee9930a036b8ac4', '8afda0e3a6cef2260c0036302a16ff4f3cb526f8726c3b007ee9930a036b8ac4', '8afda0e3a6cef2260c0036302a16ff4f3cb526f8726c3b007ee9930a036b8ac4'),
    'constraint_regions_round26.py': ('5c7e88771fe6d47c5605b7ffa7030a234312d5766c778cdcda677a7e3eff1a81', '5c7e88771fe6d47c5605b7ffa7030a234312d5766c778cdcda677a7e3eff1a81', '0c418911bbedc9a9f301dc2a83e3f1c0b9d838006442514c7d4857a8ef91f304'),
    'constraint_relations.py': ('15cfb0d11c15ff03b1a83c3e1fae381f5c796a8e218d841e8b7a4f72f5d126f0', '15cfb0d11c15ff03b1a83c3e1fae381f5c796a8e218d841e8b7a4f72f5d126f0', '15cfb0d11c15ff03b1a83c3e1fae381f5c796a8e218d841e8b7a4f72f5d126f0'),
    'constraint_repair_gold_round26.py': ('7d99679e8aa9c860a02afc4e1c37b90774c622882d01ca3cf7358da83490da02', 'd22fcc9530e87c142d28579d4d606ecccf1708318903b66f469133bc24984f09', '35f49081cc151c69d4cacff5d03175996b5856aca5adbf20875f5b526abbd9f0'),
    'constraint_repair_resume_round25.py': ('6fbd361b87b861de2ccd31036637d5e66c5780e011cc3b5e61fcb8c75fb25b99', 'bf44954f046b2dbc0140d2b426f8c58634ba1d232693887691b1fc02c993ba85', 'd401492a327afff88b8f0894e0e42dd765a97926f4eabd977995a63428b602e2'),
    'constraint_repair_round24.py': ('c80331d5f3ebf81536a16d574db95eebe39bdc84d3a9889915ae54d0ee9087c7', 'c80331d5f3ebf81536a16d574db95eebe39bdc84d3a9889915ae54d0ee9087c7', '4dec58bffba2f0c82527fad2d9473a2348d6660af0bd3700d85cb02c15503a72'),
    'constraint_repair_service_round26.py': ('8b31c913c7445fe388f85a49d63ddacc1b06c177319faf180455600b2ff69cdf', 'e89b54451737174811f0e2a5bd263e16079e8b23b7727518127299e26bd817c4', '380fe2e8f4789c9a2b4be4f86a9e931984e1356eb89c15f5074e5a42d0183604'),
    'constraint_separator.py': ('e8394186a1d202422ea838faf2e3628525c825e420980db40b347b2accc3c385', 'e8394186a1d202422ea838faf2e3628525c825e420980db40b347b2accc3c385', 'e8394186a1d202422ea838faf2e3628525c825e420980db40b347b2accc3c385'),
    'constraint_symbolic_work.py': ('84dacda63ca5ebb1ba31eed57d1d6aa581264f20da0b208c805d7327a4dd9841', '84dacda63ca5ebb1ba31eed57d1d6aa581264f20da0b208c805d7327a4dd9841', '84dacda63ca5ebb1ba31eed57d1d6aa581264f20da0b208c805d7327a4dd9841'),
    'constraint_symmetry.py': ('83875a6bfd298424e85639286a3bf9f158b45e19f24b6b56e2725c8bb09df153', '83875a6bfd298424e85639286a3bf9f158b45e19f24b6b56e2725c8bb09df153', '83875a6bfd298424e85639286a3bf9f158b45e19f24b6b56e2725c8bb09df153'),
    'constraint_work.py': ('8cbf834fc1e89d13e23522becfc7378bd870c6f12b46ee9b97450e0c08862bbf', '8cbf834fc1e89d13e23522becfc7378bd870c6f12b46ee9b97450e0c08862bbf', '8cbf834fc1e89d13e23522becfc7378bd870c6f12b46ee9b97450e0c08862bbf'),
    'geometry.py': ('fb34df600b516c545a8417f6ba3c11bed27dabd9b0b4531b1e4b443d6d434538', 'fb34df600b516c545a8417f6ba3c11bed27dabd9b0b4531b1e4b443d6d434538', 'fb34df600b516c545a8417f6ba3c11bed27dabd9b0b4531b1e4b443d6d434538'),
    'gpu_hybrid_resume_round25.py': ('1f3355aa34d0dd0f4df9713dbbaabca8d887a004deb5cb85a7bb364ddef0f525', '2113de18bd8aeba663f0d693de842faf6c3600e37f8ed55b92d7e88d655727c0', 'f82ac66caf9d6959c7ca9cddfcdcba5e830c12f239f0a96afd102d955093da5f'),
    'gpu_sampling.py': ('4f8e26b1290f3ac4c5df49c959f83d75d84dd3de606193586b5201315e8c21b0', '6a1a0b5763f995fdfb681958c504d21750eb9dab6073524e3051516efa3b5e68', 'd55db6fccf05ea980c52f70896db69010a360e98b3251900dc62ff8a4ead24a6'),
    'gpu_sampling_resume_round25.py': ('119c050605e99346ad0e9fe7a99d76753c09e4e04409409e81dec1acdc75da1c', 'f154a702fac63d2b8700815d9807c858bbaf5f817e0d234fd84ecc57a7bb7b98', 'f174bc5ca39f20896fbd463bdbca22539b6006490242b9fd09f53aff236a1cb8'),
    'gpu_sampling_state_round25.py': ('673e63399e323d80b630e91e5f393961033f5fe39f3ebc7fa6c9526ad194183b', '145dffcfbe630d77c748bbaecc91d458d243e21b18151ce3f5c1a2f0b39564c1', 'eba65d5847195873558fd4e9632084574dfc368e19748dcd9e4ae5058c0e8691'),
    'io_utils.py': ('9a794d24f8db1c1497635717879bb9b799f513bb9bd8bc10e51cf4d3c67b49c9', '9a794d24f8db1c1497635717879bb9b799f513bb9bd8bc10e51cf4d3c67b49c9', '9a794d24f8db1c1497635717879bb9b799f513bb9bd8bc10e51cf4d3c67b49c9'),
    'sampling_crossover_round25.py': ('37beec945027144fb62a5c3be513967d5fad9d182bec3e4ad07edd94b277416c', 'ddbb92d6da115f3d93a2694a153bbd84a3ab891f5bedf20437503cd6d7ae92b4', 'ddbb92d6da115f3d93a2694a153bbd84a3ab891f5bedf20437503cd6d7ae92b4'),
    'sampling_crossover_round26.py': ('a4fe4bff5638005fef1f2d3af911f718eec4d0c569a711f3a6d3fb7d5848bb96', '4895892f124c9f6547dcaddddea76b500b2ef6f74a5e86e0145798bbcec8242f', 'b365dda6955078b4b272708bfef25e534ba2e63150de5747d187fbbb483c3b83'),
    'search_options.py': ('a031ac39552a9f028191a4b62ff29a096777c7e3bbc671c0fd0ced4496b00e81', '18455f7a07bf56e1e2364d24f67ac1dfd222a3420a6bea8003dbdaa8ac81eaca', '7eef5cb6fd64667eb75f42abfa65c7aaa80a6a2b59cb3cc18774451d494791ac'),
    'search_problem.py': ('ee3447ba79ce73aebabe6b3f9eea8c479e5333fff71af171f5e3d15c4d1ea7f1', 'ee3447ba79ce73aebabe6b3f9eea8c479e5333fff71af171f5e3d15c4d1ea7f1', 'ee3447ba79ce73aebabe6b3f9eea8c479e5333fff71af171f5e3d15c4d1ea7f1'),
    'validator.py': ('f2c200e4f3e6d9dec5addd52f411bb8d8e60c795b5422ce971d2e99d5eb96828', 'f2c200e4f3e6d9dec5addd52f411bb8d8e60c795b5422ce971d2e99d5eb96828', 'f2c200e4f3e6d9dec5addd52f411bb8d8e60c795b5422ce971d2e99d5eb96828'),
}

_R28 = {
    'constraint_lns_resume_round25.py': ('5c42f53ce0f5fad6485e651d13e0e7cf44d47ad065ea0a0c77b8e22f3a707ed8', 'e18dcf659c75ccf746f5d0bbd3f6acd1263adee7a60a414e4fb8dc2d3f7007e4', '69b5b001153b65386f86b731134d704498cbee104a4cdbfc3d8a1c15e5f30b25'),
    'constraint_symbolic_frontier.py': ('e0a2e06c438d2141fae436b2169186165eb22beb21a614a866be366ed8a713df', '864ee26bce5911c853b1e1fba4d2cc897e110daf2285ae90a0b6594d6596c0b6', '04bbcb149c52552dbba78e6914aa39f6a558022ba8083bb7285b1a8469e42429'),
    'gpu_domain_round24.py': ('56018ee01984e57def7a6f100467a9157442a026d1146e74b95c8837a107e89f', 'd6aeb828dd23613ed00e61979a5b41dd79bf4da16fa69aac7dfc80d532b03e86', 'c28ef46d5a6ac0f07505c93e34f90e920e1343e987815ddeb62b9c70c953f063'),
    'gpu_domain_runner.py': ('584eb134831acb3a63f26bfa950133b36e634e5bf95e8700b12c35cd774fe1ab', '2dc06201e7bc9d360dc4d03ba6503065fe34fa1f04cec46cc63fa7741190bb31', '420fe3066d71165764d6f9de856079084aeda1466b4b0bdea5534751e91b2753'),
    'domain_search_round24.cu': ('68bdf1b64bd56540454b26cc4f8329da9749cdc15d7a09fc590a6c7c7b091486', 'c42e0e0d661209e25c7c60190f8ae75d2b608b962f2fa3edcf394aa075d28508', '84ccf676f549af30646bbb4b6f59a1f9ca2be28223ff9a51834657c57accb984'),
}

# The heuristic controllers use the original pruning relation. These two source
# changes introduce opt-in Gold DFS capabilities and preserve heuristic state.
_R32_HEURISTIC = {
    'constraint_pruning.py': ('69ea908ccf4b2cdafe409413cbe3eaebed8d9765c6a544ebd31ae7071bdc8b4a', 'c64363073be2fca374ca688917fcb35785a44da326baea6c0bcb9e7d452d41e3', 'f5abf9067e9f5894f99b8b49983cedc384ece47c1b15d9e3885ffdbe1fd1b8d9'),
    'search_problem.py': ('42514de3bc076252a2a1b3a4992bb9f4cd5b8a647f47a2e8c24f8fc62a92d3aa', 'ee3447ba79ce73aebabe6b3f9eea8c479e5333fff71af171f5e3d15c4d1ea7f1', 'a057bf686d41ce6ffa4dfba383bc86d08e1a9092fd1ccea9438bc08754c2d0d6'),
}

# In these heuristic bindings only digest() is used from constraint_work,
# and the physical Gold filters are dormant. Each alias is pinned to its
# reviewed current LF image and both text layouts of the v1.14.0 image.
_R34_HEURISTIC = {
    # The opt-in GPU-domain partition policy has no path into these heuristic controllers.
    'search_options.py': ('9bc5ad2f1b2e2c039dd18d141b52693419f2f1118260869c32e673bac869981c',
                          'a031ac39552a9f028191a4b62ff29a096777c7e3bbc671c0fd0ced4496b00e81',
                          'cf8b4d9d6bea962ea4078f64cdf53662713e682bf3783abce054500cf1346d5d'),
    'constraint_work.py': ('a7804dfb62f7585ebe0ef578a86968646c0bc0ebdca093d811452832d7a8f48c',
                           '8cbf834fc1e89d13e23522becfc7378bd870c6f12b46ee9b97450e0c08862bbf',
                           '05f93d66eca7d55804c720de4523d7e090125795ab56de55421af9f1ecf87c88'),
    'r27_structure_filters.py': ('a82b3c2132db0161500542b2b6a7e8de097b46bbb4fefb141955e7fa3d457e93',
                                 '42836dae0bc4e2fa10541826a56861b2c9319c7b8cdaa9b522fd6b3cd9d8f361',
                                 'dd287363636dc01e0e96f86b9cd41bb383446f71c1a3584de03dceb4e616a377'),
    'r32_gold_bonds.py': ('4dea3c29ec1af1a8cd4d911c835baa583ae7e2606590b4bd7b87b4f714ec403f',
                          '77a82d49d811d97476e8cb4a707f50804abc79a7f8907279e1eba6d7c6ff4cae',
                          '1daaf2a03f103f2542fb149bd26769499097cc63126c72a910df33d4414c8263'),
}
# Only digest() from this file participates in the historical heuristic controller.
_R35_HEURISTIC = {
    'constraint_work.py': ('7d7ee75b416c997751a950a5254a083cc650e25040d5ff73af1c2b0f21ce70e9',
                           'a7804dfb62f7585ebe0ef578a86968646c0bc0ebdca093d811452832d7a8f48c',
                           'd04c92c4d207ad17f095554f3e163f105c4a31ceea716491194fe61bfbb638c5'),
}
_R34_OPTIONAL = {
    'r34_coverage_index.py': 'a75b25eb32f5213a709da2b4a3b909a520f405c4d2751ea1fc5e906b9df51efa',
}
_PROFILES = ('v115-git','v115-lf','v115-crlf','v114-git','v114-lf','v114-crlf','lf','windows','v112-lf','v112-crlf')


# Bind each historical text image to the exact current image reviewed here.
# Git checkouts and release archives can use LF or CRLF for the same source.
@lru_cache(maxsize=256)
def _variants(name,current):
    path=Path(__file__).parent/name
    if Path(name).name!=name or not path.is_file():
        return ()
    raw=path.read_bytes()
    lf=raw.replace(b'\r\n',b'\n')
    crlf=lf.replace(b'\n',b'\r\n')
    variants={sha256(data).hexdigest() for data in (raw,lf,crlf)}
    if current not in variants:
        return ()
    for reviewed in (_R35_HEURISTIC, _R34_HEURISTIC, _R32_HEURISTIC):
        heuristic = reviewed.get(name)
        if heuristic and heuristic[0] in variants:
            variants.update(heuristic[1:])
    version = _R28.get(name)
    if version and version[0] in variants:
        variants.update(version[1:])
    old_new=_COMPATIBLE.get(name)
    if old_new and old_new[1] in variants:
        variants.update((old_new[0],_V17_CRLF[name]))
    return variants


# Transitive import discovery now sees optional modules that the released
# heuristic controllers did not execute. Only omit these exact reviewed bytes
# when reconstructing an older heuristic source manifest.
_R32_OPTIONAL = {
    'r27_structure_filters.py': '42836dae0bc4e2fa10541826a56861b2c9319c7b8cdaa9b522fd6b3cd9d8f361',
    'r32_gold_bonds.py': '77a82d49d811d97476e8cb4a707f50804abc79a7f8907279e1eba6d7c6ff4cae',
}

def _old_extra(name, current):
    expected = _R34_OPTIONAL.get(name) or _R32_OPTIONAL.get(name)
    return expected is not None and expected in _variants(name,current)


def compatible_sources(saved,current):
    return (isinstance(saved,dict) and saved.keys() <= current.keys()
            and (not (_R32_OPTIONAL.keys() & (current.keys()-saved.keys()))
                 or not (_R32_OPTIONAL.keys() & saved.keys()))
            and all(_old_extra(name,current[name]) for name in current.keys()-saved.keys())
            and all(isinstance(saved[name],str) and
                    (saved[name]==current[name] or saved[name] in _variants(name,current[name]))
                    for name in saved))

def compatible_identity(saved,current,legacy):
    if not isinstance(saved,str):
        return False
    return saved==current or any(saved==legacy(profile) for profile in _PROFILES)


def legacy_sources(current,profile):
    """Reconstruct only a checksum-verified published source manifest."""
    if profile not in _PROFILES:
        raise ValueError('Unknown release source layout.')
    result={}
    for name,sha in current.items():
        if (name in _R34_OPTIONAL and not profile.startswith('v115-')
                or name in _R32_OPTIONAL and not profile.startswith(('v114-','v115-'))):
            if not _old_extra(name,sha):
                return None
            continue
        if profile.startswith(('v112-','v114-','v115-')):
            variants = _variants(name,sha)
            if not variants:
                return None
            # The v1.14 Git source/archive has LF Python and a CRLF native C file.
            crlf = profile.endswith('-crlf') or profile=='v114-git' and name=='constraint_native.c'
            row = (_R35_HEURISTIC if profile.startswith('v115-') else _R34_HEURISTIC).get(name)
            if row is None and profile.startswith('v112-'):
                row = _R32_HEURISTIC.get(name)
            if row:
                if row[0] not in variants:
                    return None
                result[name] = row[2 if crlf else 1]
            else:
                body = (Path(__file__).parent/name).read_bytes().replace(b'\r\n',b'\n')
                result[name] = sha256(body.replace(b'\n',b'\r\n') if crlf else body).hexdigest()
            continue
        row=_RELEASE_TEXT.get(name)
        if row is None or row[0] not in _variants(name,sha):
            return None
        result[name]=row[1 if profile=='lf' else 2]
    return result
