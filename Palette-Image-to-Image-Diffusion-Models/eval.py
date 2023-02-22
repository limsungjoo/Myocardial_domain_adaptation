import argparse
from cleanfid import fid
from core.base_dataset import BaseDataset
from models.metric import inception_score

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--src', type=str, help='Ground truth images directory')
    parser.add_argument('-d', '--dst', type=str, help='Generate images directory')
   
    ''' parser configs '''
    args = parser.parse_args()

    # fid_score = fid.compute_fid(args.src, args.dst)
    is_mean, is_std,mae = inception_score(BaseDataset(args.dst), cuda=True, batch_size=1, resize=True, splits=10)
    
    # print('FID: {}'.format(fid_score))
    print('{} {} {}'.format(is_mean, is_std,mae))