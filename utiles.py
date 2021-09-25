from os import path as osp
import random
from multiprocessing import Pool

def scandir(dir_path, suffix=None, recursive=False, full_path=False):
    """Scan a directory to find the interested files.
    Args:
        dir_path (str): Path of the directory.
        suffix (str | tuple(str), optional): File suffix that we are
            interested in. Default: None.
        recursive (bool, optional): If set to True, recursively scan the
            directory. Default: False.
        full_path (bool, optional): If set to True, include the dir_path.
            Default: False.
    Returns:
        A generator for all the interested files with relative pathes.
    """

    if (suffix is not None) and not isinstance(suffix, (str, tuple)):
        raise TypeError('"suffix" must be a string or tuple of strings')

    root = dir_path

    def _scandir(dir_path, suffix, recursive):
        for entry in os.scandir(dir_path):
            if not entry.name.startswith('.') and entry.is_file():
                if full_path:
                    return_path = entry.path
                else:
                    return_path = osp.relpath(entry.path, root)

                if suffix is None:
                    yield return_path
                elif return_path.endswith(suffix):
                    yield return_path
            else:
                if recursive:
                    yield from _scandir(entry.path, suffix=suffix, recursive=recursive)
                else:
                    continue

    return _scandir(dir_path, suffix=suffix, recursive=recursive)

def extract_subimages():
    opt = {}
    opt['n_thread'] = 20
    # HR images
    opt['input_folder'] = 'dataset/DIV2K/DIV2K_train_HR'
    opt['save_folder'] = 'dataset/DIV2K/DIV2K_train_HR_sub'
    opt['crop_size'] = 384
    input_folder = opt['input_folder']
    save_folder = opt['save_folder']

    img_list = list(scandir(input_folder, full_path=True))
    pbar = tqdm(total=len(img_list), unit='image', desc='Extract')
    pool = Pool(opt['n_thread'])
    for path in img_list:
        pool.apply_async(worker, args=(path, opt), callback=lambda arg: pbar.update(1))
    pool.close()
    pool.join()
    pbar.close()
    print('All processes done.')

def worker(path, opt):
    crop_size = opt['crop_size']
    img_name, extension = osp.splitext(osp.basename(path))

    import cv2
    img = cv2.imread(path)

    h, w = img.shape[0:2]
    h_space = [random.randint(0,h-crop_size) for i in range(100)]
    w_space = [random.randint(0, w - crop_size) for i in range(100)]
    index = 0
    for i in range(100):
        index += 1
        cropped_img = img[h_space[i]:h_space[i] + crop_size, w_space[i]:w_space[i] + crop_size, ...]
        # cropped_img = np.ascontiguousarray(cropped_img)
        cv2.imwrite(
            osp.join(opt['save_folder'], f'{img_name}_s{index:04d}{extension}'), cropped_img)
    process_info = f'Processing {img_name} ...'
    return process_info


if __name__ == '__main__':
    extract_subimages()
