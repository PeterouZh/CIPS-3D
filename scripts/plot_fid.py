import pickle
import os

from tl2.proj.argparser import argparser_utils
from tl2.proj.matplot import plt_utils


def main(data_pkl,
         data_key,
         title,
         outdir):

  os.makedirs(outdir, exist_ok=True)

  with open(data_pkl, 'rb') as f:
    loaded_data = pickle.load(f)
    data_dict = loaded_data['FID_r64']

  data = data_dict[data_key]

  fig, ax = plt_utils.get_fig_ax()

  ax.plot(data[:, 0], data[:, 1])

  plt_utils.ax_set_ylim(ax, [0, 100])
  plt_utils.ax_set_xlabel(ax, xlabel='Iters')
  plt_utils.ax_set_ylabel(ax, ylabel='FID')
  plt_utils.ax_set_title(ax, title=title, fontsize=20)

  plt_utils.savefig(saved_file=f"{outdir}/{data_key}.png", fig=fig, debug=True)

  pass





if __name__ == '__main__':
  """
  python scripts/plot_fid.py --data_key ffhq_r64
  
  """

  parser = argparser_utils.get_parser()

  argparser_utils.add_argument_str(parser, 'data_pkl', default="datasets/data/ffhq_fid.pkl")
  argparser_utils.add_argument_str(parser, 'data_key', default="ffhq_r64")
  argparser_utils.add_argument_str(parser, 'title', default=r"FFHQ $64\times64$")
  argparser_utils.add_argument_str(parser, 'outdir', default="results/plot_fid")


  args, _ = parser.parse_known_args()
  argparser_utils.print_args(args)

  main(**vars(args))