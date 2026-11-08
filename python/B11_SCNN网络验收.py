"""网络形状、自动微分、保存重载及清单隔离验收。"""
from pathlib import Path
import json
import tempfile
import unittest
import numpy as np
import torch

from B11_SCNN网络 import SCNNB11, SameConv1d, load_scnn
from B11_SCNN训练工具 import load_split, balanced_loss, regression_metrics
from B11_数据工具 import write_json

ROOT = Path(__file__).resolve().parents[1]


class NetworkTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(77)
        torch.set_num_threads(2)

    def test_shape_and_length(self):
        model=SCNNB11().eval()
        x=torch.rand(7,1,50)*1000
        self.assertEqual(tuple(model(x).shape),(7,1))
        self.assertEqual(tuple(model.features(x).shape),(7,3,50))

    def test_invalid_shape(self):
        with self.assertRaises(ValueError): SCNNB11()(torch.rand(2,2,50))

    def test_even_kernel_padding_against_manual(self):
        layer=SameConv1d(1,1,4)
        with torch.no_grad():
            layer.conv.weight.fill_(1);layer.conv.bias.zero_()
        x=torch.arange(1.,6.).reshape(1,1,5)
        np.testing.assert_allclose(layer(x).detach().numpy().ravel(),[6,10,14,12,9])

    def test_balanced_loss_ignores_class_count(self):
        p=torch.tensor([[1.],[2.]])*1000
        y=torch.tensor([[0.],[1.]])*1000
        self.assertAlmostEqual(float(balanced_loss(p,y)),1)
        pr=torch.cat([p[:1].repeat(100,1),p[1:]])
        yr=torch.cat([y[:1].repeat(100,1),y[1:]])
        self.assertAlmostEqual(float(balanced_loss(pr,yr)),1)

    def test_loss_autograd_gradcheck(self):
        x=torch.tensor([[100.],[20.],[1400.]],dtype=torch.double,requires_grad=True)
        y=torch.tensor([[50.],[0.],[1100.]],dtype=torch.double)
        self.assertTrue(torch.autograd.gradcheck(lambda a: balanced_loss(a,y),(x,),eps=1e-4))

    def test_serialization_exact(self):
        model=SCNNB11().eval();x=torch.rand(3,1,50)*1000
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'weights.pth';torch.save(model.state_dict(),p)
            torch.testing.assert_close(model(x),load_scnn(p)(x),rtol=0,atol=0)

    def test_loaded_split_and_baselines(self):
        p=ROOT/'data/B11_SCNN滑窗样例/B11_SCNN滑窗清单.json'
        x,y,ids=load_split(p,'train')
        self.assertEqual(len(x),17424)
        self.assertTrue(all(v.startswith('B11_train_') for v in ids))
        result=regression_metrics(y.numpy(),y.numpy())
        self.assertEqual(result['mse'],0)

    def test_cross_split_rejected_before_files(self):
        p=ROOT/'data/B11_SCNN滑窗样例/B11_SCNN滑窗清单.json'
        spec=json.loads(p.read_text(encoding='utf-8'))
        train=next(r for r in spec['records'] if r['split']=='train')
        val=next(r for r in spec['records'] if r['split']=='val')
        val['source_scene_id']=train['source_scene_id']
        with tempfile.TemporaryDirectory() as td:
            fake=Path(td)/'manifest.json';write_json(fake,spec)
            with self.assertRaisesRegex(ValueError,'跨集合'): load_split(fake,'train')


if __name__=='__main__':
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(NetworkTests))
    write_json(ROOT/'results/B11_SCNN训练/B11_网络单元验收.json',{
        'tests_run':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),
        'passed':result.wasSuccessful(),'scope':'网络实现和数据接口；不作为真实SAR性能结论'})
    raise SystemExit(not result.wasSuccessful())
