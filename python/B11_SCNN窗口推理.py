"""加载SCNN权重，对已规范化的一维窗口推理并还原中心幅度。"""
import argparse
from pathlib import Path
import numpy as np
import torch
from scipy.io import savemat

from B11_SCNN网络 import load_scnn
from B11_SCNN训练工具 import predict_batches
from B11_SCNN数据接口 import restore_centers

ROOT=Path(__file__).resolve().parents[1]


def infer(input_path, weights, output_path):
    torch.set_num_threads(2)
    with np.load(input_path,allow_pickle=False) as sample:
        x=sample['input'].copy()
        scales=sample['scale'].copy()
        coordinates={key:sample[key].copy() for key in ('axis','line','center_original')}
    if x.ndim!=3 or x.shape[1:]!=(1,50) or not np.isfinite(x).all() or np.any(x<0):
        raise ValueError('输入必须为有限非负幅度窗口 [N,1,50]')
    if scales.shape!=(len(x),) or any(v.shape!=(len(x),) for v in coordinates.values()):
        raise ValueError('尺度或坐标与窗口数量不符')
    model=load_scnn(weights,'cpu')
    raw=predict_batches(model,torch.from_numpy(x.astype(np.float32))).numpy()
    nonnegative=np.maximum(raw,0)
    restored=restore_centers(nonnegative,scales)
    result={'prediction_raw':raw,'prediction_nonnegative':nonnegative,
            'center_amplitude':restored,'scale':scales,**coordinates}
    output_path.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(output_path,**result)
    savemat(output_path.with_suffix('.mat'),result,do_compression=True)
    print(f'已导出 {len(x)} 个窗口的中心幅度；这是分散中心点，不是重建后的整幅影像。')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--weights',type=Path,default=ROOT/'weights/B11_SCNN.pth')
    parser.add_argument('--output',type=Path,default=ROOT/'results/B11_SCNN训练/B11_窗口推理示例.npz')
    args=parser.parse_args()
    infer(args.input,args.weights,args.output)
