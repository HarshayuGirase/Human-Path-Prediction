import yaml
import argparse
import IPython
from IPython import embed

parser = argparse.ArgumentParser(description='Config Path')
parser.add_argument('--filename', '-fn', type=str, default='default.yaml')
parser.add_argument('--change_optimal', '-co', action='store_true')
args = parser.parse_args()

if not args.change_optimal and args.filename == 'optimal.yaml':
    raise Exception("Please choose a different filename or use '-co' arg if changing optimal config.")
elif args.change_optimal:
    args.filename = 'optimal.yaml'

contents = {}

contents['dec_size'] = [1024,512,1024]
contents['enc_dest_size'] = [8,16]
contents['enc_latent_size'] = [8,50]
contents['enc_past_size'] = [512,256]
contents['predictor_hidden_size'] = [1024,512,256]
contents['non_local_theta_size'] = [256,128,64]
contents['non_local_phi_size'] = [256,128,64]
contents['non_local_g_size'] = [256,128,64]
contents['non_local_dim'] = 128
contents['adl_reg'] = 1
contents['kld_reg'] = 1
contents['fdim'] = 16
contents['zdim'] = 16
contents['learning_rate'] = 0.0003
contents['num_epochs'] = 650
contents['nonlocal_pools'] = 3
contents['n_values'] = 20
contents['mu'] = 0
contents['sigma'] = 1.3

contents['dataset_type'] = 'image'
contents['data_scale'] = 1.86
contents['gpu_index'] = 0
contents['normalize_type'] = 'shift_origin'
contents['num_workers'] = 0
contents['dist_thresh'] = 100
contents['time_thresh'] = 0
contents['train_b_size'] = 512
contents['test_b_size'] = 4096

contents['past_length'] = 8
contents['future_length'] = 12

with open(args.filename, 'w') as file:
    yaml.dump(contents, file, sort_keys=True)
file.close()
