import glob,os,cv2,onnxruntime as ort,numpy as np
import argparse
parser=argparse.ArgumentParser()
parser.add_argument('--root', default='.')
parser.add_argument('--onnx', required=True)
parser.add_argument('--board-dir', required=True)
parser.add_argument('--images', required=True)
parser.add_argument('--limit', type=int, default=20)
args=parser.parse_args()
root=os.path.abspath(args.root); ims=os.path.abspath(args.images); fp=os.path.abspath(args.onnx); bd=os.path.abspath(args.board_dir)
s=ort.InferenceSession(fp,providers=['CPUExecutionProvider']); fs=sorted(glob.glob(ims+'/*.jpg'))[:args.limit]
def xform(f):
 im=cv2.cvtColor(cv2.imread(f),cv2.COLOR_BGR2RGB);im=cv2.resize(im,(960,832));return np.transpose(im.astype('float32')/255,(2,0,1))[None]
def st(a,b):
 a=a.ravel().astype('float32');b=b.ravel().astype('float32');return np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)),np.mean(np.abs(a-b)),np.sqrt(np.mean((a-b)**2))
cs=[]; ss=[]; ag=[]; ii=[]
for f in fs:
 n=os.path.basename(f).split('.')[0]; a=s.run(None,{'images':xform(f)}); d=np.fromfile(f'{bd}/{n}_0.bin','float32').reshape(a[0].shape); m=np.fromfile(f'{bd}/{n}_1.bin','float32').reshape(a[1].shape)
 cs.append(st(a[0],d)); ss.append(st(a[1],m)); p=a[1]>0;q=m>0; ag.append(np.mean(p==q)); inter=np.logical_and(p,q).sum(); uni=np.logical_or(p,q).sum();ii.append(inter/uni if uni else 1.)
print('det cosine/mae/rmse',np.mean(cs,axis=0));print('seg cosine/mae/rmse',np.mean(ss,axis=0));print('seg threshold agreement',np.mean(ag),'iou',np.mean(ii));print('seg per image iou',ii)
