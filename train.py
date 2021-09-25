from SwinIC import *


# differentiable rounding function
class BypassRound(Function):
    @staticmethod
    def forward(ctx, inputs):
        return torch.round(inputs)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output

bypass_round = BypassRound.apply
class Logger(object):
    '''Save training process to log file with simple plot function.'''
    def __init__(self, fpath, title=None, resume=False):
        self.file = None
        self.resume = resume
        self.title = '' if title == None else title
        if fpath is not None:
            if resume:
                self.file = open(fpath, 'r')
                name = self.file.readline()
                self.names = name.rstrip().split('\t')
                self.numbers = {}
                for _, name in enumerate(self.names):
                    self.numbers[name] = []

                for numbers in self.file:
                    numbers = numbers.rstrip().split('\t')
                    for i in range(0, len(numbers)):
                        self.numbers[self.names[i]].append(numbers[i])
                self.file.close()
                self.file = open(fpath, 'a')
            else:
                self.file = open(fpath, 'w')

    def set_names(self, names):
        if self.resume:
            pass
        # initialize numbers as empty list
        self.numbers = {}
        self.names = names
        for _, name in enumerate(self.names):
            self.file.write(name)
            self.file.write('\t')
            self.numbers[name] = []
        self.file.write('\n')
        self.file.flush()


    def append(self, numbers):
        assert len(self.names) == len(numbers), 'Numbers do not match names'
        for index, num in enumerate(numbers):
            self.file.write("{0:.6f}".format(num))
            self.file.write('\t')
            self.numbers[self.names[index]].append(num)
        self.file.write('\n')
        self.file.flush()

    def plot(self, names=None):
        names = self.names if names == None else names
        numbers = self.numbers
        for _, name in enumerate(names):
            x = np.arange(len(numbers[name]))
            plt.plot(x, np.asarray(numbers[name]))
        plt.legend([self.title + '(' + name + ')' for name in names])
        plt.grid(True)

    def close(self):
        if self.file is not None:
            self.file.close()


def concat_images(image1, image2):
	"""
	Concatenates two images together
	"""
	result_image = Image.new('RGB', (image1.width + image2.width, image1.height))
	result_image.paste(image1, (0, 0))
	result_image.paste(image2, (image1.width, 0))
	return result_image


class Preprocess(object):
    def __init__(self):
        pass
    def __call__(self, PIL_img):
        img = torch.from_numpy(np.asarray(PIL_img, dtype=np.float32).transpose((2, 0, 1)))
        img /= 127.5
        img -= 1.0
        return img

@torch.no_grad()
def test(net,epoch,val_data,val_loader,val_transform,criterion):
    start_time = time.time()
    net.eval()
    list_test_v_loss = 0.
    list_test_v_mse = 0.
    list_test_v_ms_ssim = 0.
    list_test_bpp_z = 0.
    list_test_bpp_y = 0.
    list_test_v_psnr = 0.
    cnt = 0
    with tqdm(total=len(val_data), desc=f'Epoch test {epoch+1}/{args.epoch}', unit='img',ncols=100) as pbar:
        for i, (images,fn) in enumerate(val_loader):
            filename = fn[0].split('/')[-1].split('.')[0]
            images = torch.stack([image.cuda() for image in images], dim=0)
            _, _, h_old, w_old = images.shape
            num_pixels = (h_old * w_old)
            # window_size = 64
            # todo：修改targets mask 大小与图片一致 需要缩放bbox和mask maskrcnn自带resize需要重写
            h_pad = (h_old // (16*args.windowsize)) * (16*args.windowsize) if h_old % (16*args.windowsize) == 0 else (h_old // (16*args.windowsize) + 1) * (16*args.windowsize)
            w_pad = (w_old // (16*args.windowsize)) * (16*args.windowsize) if w_old % (16*args.windowsize) == 0 else (w_old // (16*args.windowsize) + 1) * (16*args.windowsize)
            images_pad = torch.cat([images, torch.flip(images, [2])], 2)[:, :, :h_pad, :]
            images_pad = torch.cat([images_pad, torch.flip(images_pad, [3])], 3)[:, :, :, :w_pad]
            x_hat,y_hat, z_hat,means,variances,probs,probs_lap,probs_log,probs_mix = net(images_pad, mode='eval')
            x_hat = x_hat[..., :h_old, :w_old]

            images = torch.round(images * 255).float()
            x_hat = torch.round(torch.clamp(x_hat * 255, 0, 255)).float()
            v_loss, v_mse, v_ms_ssim, latent_rate, hyperlatent_rate = criterion(images, x_hat, y_hat, z_hat, means, variances,
                                                                          probs, probs_lap, probs_log, probs_mix, lmbda,
                                                                          num_pixels, args.model_type)

            # 保存样张
            if filename == 'kodim01':
                reconstructed_image = torchvision.transforms.ToPILImage(mode='RGB')(x_hat.to('cpu')[0]/255)
                # image = torchvision.transforms.ToPILImage(mode='RGB')(images.to('cpu')[0]/255)
                # result_image = concat_images(image, reconstructed_image)
                reconstructed_image.save("test_images/{}{}qp{}{}{}{}{}{}.png".format(epoch,args.model_type,args.qp, filename,args.patchsize,channel,args.windowsize,args.date))

            v_psnr = torch.mean(20 * torch.log10(255 / torch.sqrt(v_mse)), 0)
            bpp_y = latent_rate
            bpp_z = hyperlatent_rate
            
	    list_test_v_loss += v_loss.item()
            list_test_v_mse += v_mse.item()
            list_test_v_psnr += v_psnr.item()
            list_test_v_ms_ssim += v_ms_ssim.item()
            list_test_bpp_y += bpp_y.item()
            list_test_bpp_z += bpp_z.item()
            cnt += 1
            pbar.update(images.shape[0])


    batch_val_mse = list_test_v_mse / cnt
    batch_val_ms_ssim = list_test_v_ms_ssim/cnt
    batch_val_psnr = list_test_v_psnr / cnt
    batch_val_bpp_y = list_test_bpp_y / cnt
    batch_val_bpp_z = list_test_bpp_z / cnt
    batch_val_bpp_real = batch_val_bpp_y+batch_val_bpp_z
    val_loss = list_test_v_loss/cnt
    timestamp = time.time()
    print('[Epoch %04d TEST %.1f seconds] v_loss: %.4e v_mse: %.4e v_ms_ssim: %.4e v_psnr: %.4e bpp: %.4e bpp_y: %.4e bpp_z: %.4e' % (
        epoch,
        timestamp - start_time,
        val_loss,
        batch_val_mse,
        batch_val_ms_ssim,
        batch_val_psnr,
        batch_val_bpp_real,
        batch_val_bpp_y,
        batch_val_bpp_z,
    ))
    net.train()

    return val_loss,batch_val_psnr, batch_val_mse,batch_val_ms_ssim,batch_val_bpp_real,batch_val_bpp_y, batch_val_bpp_z


def listdir(path, list_name):
    for file in os.listdir(path):
        file_path = os.path.join(path, file)
        list_name.append(file_path)

class SelfDataset(torchvision.datasets.VisionDataset):
    def __init__(self, root, transform=None, target_transform=None):  # 初始化一些需要传入的参数
        super(SelfDataset, self).__init__(root)
        self.root = root
        # fh = open(root + datatxt, 'r')  # 按照传入的路径和txt文本参数，打开这个文本，并读取内容
        imgs = []
        listdir(self.root,imgs)
        self.imgs = imgs
        self.transform = transform

    def __getitem__(self, index):  # 这个方法是必须要有的，用于按照索引读取每个元素的具体内容
        fn = self.imgs[index]  # fn是图片path #fn和label分别获得imgs[index]也即是刚才每行中word[0]和word[1]的信息
        img = Image.open(fn).convert('RGB')  # 按照path读入图片from PIL import Image # 按照路径读取图片

        if self.transform is not None:
            img = self.transform(img)  # 是否进行transform
        return img,fn  # return很关键，return回哪些内容，那么我们在训练时循环读取每个batch时，就能获得哪些内容

    def __len__(self):  # 这个函数也必须要写，它返回的是数据集的长度，也就是多少张图片，要和loader的长度作区分
        return len(self.imgs)

def main():
    # todo:数据预处理，数据集下载
    num_pixels = (args.patchsize * args.patchsize)
    train_transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    train_data = SelfDataset(args.train_path,train_transform)
    train_sampler = torch.utils.data.RandomSampler(train_data)
    batch_train_sampler = torch.utils.data.BatchSampler(train_sampler, args.batchsize, drop_last=True)
    training_loader = torch.utils.data.DataLoader(
        train_data,
        num_workers=args.num_workers,
        batch_sampler=batch_train_sampler,
    )

    val_transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    val_data = SelfDataset(args.val_path,val_transform)
    # val_sampler = torch.utils.data.RandomSampler(val_data)
    # batch_val_sampler = torch.utils.data.BatchSampler(val_sampler, 1, drop_last=False)
    val_loader = torch.utils.data.DataLoader(
        val_data,
        num_workers=0,
        # batch_sampler=batch_val_sampler,
    )

    net = Net(lmbda,channel,args.windowsize)
    # 单卡训练
    net = nn.DataParallel(net,output_device=0)
    net.cuda()
    criterion = RateDistortionLoss()
    opt = optim.AdamW(net.parameters(), lr=args.learning_rate,weight_decay=0.05)
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=1,
                                               threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08)
    # for checkpoint resume
    st_epoch = 0
    ###################
    logger_train = Logger(
        os.path.join('result', 'type'+str(args.model_type)+'_qp'+str(args.qp)+'_log_' + str(args.batchsize) + '_' + str(args.learning_rate) + '_' + str(lmbda) + '_' + str(args.patchsize) + "_"+str(channel)+'_'+str(args.windowsize)+'_'+str(args.date)+'.txt'))
    logger_val = Logger(
        os.path.join('result',
                     'type'+str(args.model_type)+'_qp'+str(args.qp)+'_log_val_' + str(args.batchsize) + '_' + str(args.learning_rate) + '_' + str(lmbda) + '_' + str(args.patchsize) + "_"+str(channel)+'_'+str(args.windowsize)+'_'+str(args.date)+ '.txt'))
    logger_train.set_names(
        ['Epoch', 'Train Loss', 'Train X_MSE','Train MS_SSIM', 'Train Bpp','Train Bpp_y','Train Bpp_z'])
    logger_val.set_names(['Epoch','Val Loss','Val PSNR', 'Val X_MSE','Val MS_SSIM', 'Val Bpp','Val Bpp_y','Val Bpp_z'])

    val_loss_best = None
    for epoch in range(st_epoch, args.epoch):
        net.train()
        start_time = time.time()
        list_train_loss = 0.
        list_train_mse = 0.
        list_train_ms_ssim = 0.
        list_train_bpp_y = 0.
        list_train_bpp_z = 0.
        cnt = 0
        with tqdm(total=len(train_data), desc=f'Epoch train {epoch + 1}/{args.epoch}', unit='img',ncols=100) as pbar:
            for i, (images,fn) in enumerate(training_loader):
                opt.zero_grad()
                images = torch.stack([image.cuda() for image in images], dim=0)
                x_hat,y_hat, z_hat,means,variances,probs,probs_lap,probs_log,probs_mix = net(images,'train')
                images = torch.round(images * 255).float()
                x_hat = bypass_round(torch.clamp(x_hat * 255, 0, 255)).float()
                loss,mse,ms_ssim,latent_rate,hyperlatent_rate = criterion(images,x_hat,y_hat, z_hat,means,variances,probs,probs_lap,probs_log,probs_mix,lmbda,num_pixels,args.model_type)

                if np.isnan(loss.item()):
                    raise Exception('NaN in loss')
                loss.backward()
                opt.step()
                list_train_loss += loss.item()
                list_train_mse += mse.item()
                list_train_ms_ssim += ms_ssim.item()
                list_train_bpp_y += latent_rate.item()
                list_train_bpp_z += hyperlatent_rate.item()

                pbar.set_postfix(train_loss='{:.6f}'.format(loss.detach().cpu().numpy()))
                pbar.update(images.shape[0])
                cnt += 1
                # if cnt == 10:
                #     break

        batch_train_loss = list_train_loss / cnt
        batch_train_mse = list_train_mse / cnt
        batch_train_ms_ssim = list_train_ms_ssim/cnt
        batch_train_bpp_y = list_train_bpp_y / cnt
        batch_train_bpp_z = list_train_bpp_z / cnt
        batch_train_bpp = batch_train_bpp_y+batch_train_bpp_z
        timestamp = time.time()
        print('[Epoch %04d TRAIN %.1f seconds] Loss: %.4e bpp: %.4e bpp_y: %.4e bpp_z: %.4e mse: %.4e ms_ssim: %.4e' % (
            epoch, timestamp - start_time, batch_train_loss, batch_train_bpp,batch_train_bpp_y, batch_train_bpp_z,
            batch_train_mse,batch_train_ms_ssim))

        # if (epoch + 1) % 5 == 0:
        # 保存样张 并进行测试
        print('\nEpoch {}/{}] Loss: {:.4f}'.format(epoch + 1, args.epoch, loss.detach().cpu().numpy()))
        reconstructed_image = torchvision.transforms.ToPILImage(mode='RGB')(x_hat.to('cpu')[0]/255)
        image = torchvision.transforms.ToPILImage(mode='RGB')(images.to('cpu')[0]/255)
        result_image = concat_images(image, reconstructed_image)
        result_image.save("train_images/epoch{}{}qp{}{}{}{}{}.png".format(epoch, args.model_type,args.qp,args.patchsize,
                                                                        channel,args.windowsize,args.date))
        val_loss,batch_val_psnr, batch_val_mse,batch_val_ms_ssim,batch_val_bpp_real,batch_val_bpp_y, batch_val_bpp_z = test(net.module,epoch,val_data,val_loader,val_transform,criterion)
        # todo 保存训练数据
        logger_train.append(
            [epoch, batch_train_loss, batch_train_mse,batch_train_ms_ssim, batch_train_bpp, batch_train_bpp_y, batch_train_bpp_z])

        logger_val.append(
            [epoch, val_loss, batch_val_psnr, batch_val_mse, batch_val_ms_ssim, batch_val_bpp_real, batch_val_bpp_y,
             batch_val_bpp_z])
        # 降低学习率
        sch.step(val_loss)
        if val_loss_best is None or (val_loss > val_loss_best):
            val_loss_best = val_loss
            print('[INFO] Saving')
            if not os.path.isdir(args.checkpoint_dir):
                os.mkdir(args.checkpoint_dir)
                # 临时保存参数
            torch.save(net.state_dict(), './{}/SwinIC_{}qp{}_{}_{}_{}.ckpt'.format(
                       str(args.checkpoint_dir),str(args.model_type),str(args.qp),str(args.patchsize),str(args.windowsize),str(args.date)))
            torch.save(opt.state_dict(), './{}/SwinIC_opt_{}qp{}_{}_{}_{}.ckpt'.format(str(args.checkpoint_dir),str(args.model_type),str(args.qp),str(args.patchsize),str(args.windowsize),str(args.date)))



def make_dir(args):
    if not os.path.isdir(args.checkpoint_dir):
        os.mkdir(args.checkpoint_dir)
    if not os.path.isdir('result'):
        os.mkdir('result')
    if not os.path.isdir('test_images'):
        os.mkdir('test_images')
    if not os.path.isdir('train_images'):
        os.mkdir('train_images')
    if not os.path.isdir('result'):
        os.mkdir('result')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "output", nargs="?",
        help="Output filename.")
    parser.add_argument(
        "--train_path", default='dataset/DIV2K/DIV2K_train_HR_sub', type=str,
        help='train dataset path')
    parser.add_argument(
        "--val_path", default='dataset/kodak', type=str,
        help='val dataset path')
    parser.add_argument(
        "--checkpoint_dir", default="checkpoint",
        help="Directory where to save/load model checkpoints.")
    parser.add_argument(
        "--batchsize", type=int, default=16,
        help="Batch size for training.")
    parser.add_argument(
        '--gpu', default='0,1', type=str, help='gpu id')
    parser.add_argument(
        "--qp", type=int, default=1,
        help="quantization parameter")
    parser.add_argument(
        "--num_workers", type=int, default=12,
        help="num workers for data loading.")
    parser.add_argument(
        "--learning_rate", type=int, default=0.0001,
        help="learning rate")
    parser.add_argument(
        "--model_type", default=0, type=int,
        help="Model type, choose from 0:PSNR 1:MS-SSIM"
    )
    parser.add_argument(
        "--patchsize", default=384, type=int,
        help="Size of image patches for training."
    )
    parser.add_argument(
        "--epoch", default=100, type=int,
        help=""
    )
    parser.add_argument(
        "--windowsize", default=8, type=int,
        help="Size of Swin Transformer window for training."
    )
    parser.add_argument(
        "--date", default="0925",
        help="date")

    args = parser.parse_args()
    make_dir(args)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    if args.model_type == 0:
        lmbda = {"1":0.0016,"2":0.0032,"3":0.0075,"4":0.015,"5":0.023,"6":0.03,"7":0.045}[str(args.qp)]
        if args.qp <= 3:
            channel = 128
        else:
            channel = 256
    else:
        lmbda = {"1": 12, "2": 40, "3": 80, "4": 120}[str(args.qp)]
        if args.qp <= 2:
            channel = 128
        else:
            channel = 256
    main()

