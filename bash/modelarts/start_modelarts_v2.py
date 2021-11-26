import os
import sys
import argparse


def parser_args_from_list(name, argv_list, type='list'):
  """

  :param name: --tl_opts
  :param argv_list:
  :return:
  """
  print(f"Parsering {name} from \n{argv_list}")
  parser = argparse.ArgumentParser()
  if type == 'list':
    parser.add_argument(name, type=str, nargs='*', default=[])
  else:
    parser.add_argument(name)
  args, _ = parser.parse_known_args(args=argv_list)

  value = getattr(args, name.strip('-'))
  print(f"{name}={value}")
  return value

#  ['/home/ma-user/modelarts/user-job-dir/pi-GAN-exp/tl2_lib/tl2/modelarts/scripts/start_modelarts_v2.py',
#  '--bash=pwd', '&&', 'ls', '-la']
print(f"argv:\n {sys.argv}")
argv = []
for v in sys.argv:
  if v.startswith('--bash'):
    argv.extend(v.split('='))
  else:
    argv.append(v)
argv = argv[:argv.index('--bash') + 1] + [' '.join(argv[argv.index('--bash') + 1:])]
print(f"converted argv:\n {argv}")

bash = parser_args_from_list(name="--bash", argv_list=argv, type='list')

# parser = argparse.ArgumentParser()
# # parser.add_argument('--train_url', type=str, default=None, help='the output path')
# # parser.add_argument('--bash', type=str, default=None)
# parser.add_argument('--bash', nargs="+", default=[])
# args, unparsed = parser.parse_known_args()
# print(f"unparsed:\n {unparsed}")
# print(f"args:\n {args}")

command = ' '.join(bash)
print(f'The running command is: \n{command}')
os.system(command)

# /bucket-3690/ZhouPeng/codes/pi-GAN-exp/tl2_lib/tl2/modelarts/scripts/start_modelarts_v2.py





