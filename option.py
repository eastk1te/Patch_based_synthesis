import argparse

def parse_opt():
    parser = argparse.ArgumentParser()
    
    # miscellaneous arguments for training
    parser.add_argument("--image_path", type=str, default='./image/image1.jpg')
    parser.add_argument("--image_file_name", type=str, default='')
    parser.add_argument("--row_num", type=int, default=4)
    parser.add_argument("--col_num", type=int, default=4)
    parser.add_argument("--output_filename", type=str, default='./image_patches/output_image')
    parser.add_argument("--result_filename", type=str, default='./merge_patches/result')
    parser.add_argument("--random", type=int, default=0)
    parser.add_argument("--set_seed", type=bool, default=True)

    opt = parser.parse_args([])

    return opt