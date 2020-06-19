import numpy as np 
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
import ipdb
import torchvision.utils as vutils
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from model import *
from ops import *
from chord_to_emb import *

class get_dataloader(object):
    def __init__(self, data, prev_data, y):
        self.size = data.shape[0]
        self.data = torch.from_numpy(data).float()
        self.prev_data = torch.from_numpy(prev_data).float()

        # Change the float type to long if you want to use an embedding model
        self.y   = torch.from_numpy(y).float()

    def __getitem__(self, index):
        return self.data[index],self.prev_data[index], self.y[index]

    def __len__(self):
        return self.size

def load_data():
    #######load the data########
    check_range_st = 0
    check_range_ed = 129
    pitch_range = check_range_ed - check_range_st-1

    X_tr = np.load('X_tr_augmented.npy')
    prev_X_tr = np.load('prev_X_tr_augmented.npy')
    y_tr    = np.load('chord_tr_augmented.npy')
    
    # uncomment the following line if you want to use an embedding model
    #y_tr = list_chord_to_emb(y_tr)
    
    X_tr = X_tr[:,:,:,check_range_st:check_range_ed]
    prev_X_tr = prev_X_tr[:,:,:,check_range_st:check_range_ed]

    train_iter = get_dataloader(X_tr,prev_X_tr,y_tr)
    train_loader = DataLoader(train_iter, batch_size=72, shuffle=True)

    print('data preparation is completed')
    #######################################
    return train_loader


def main():
    is_train = 0
    is_draw = 0
    is_sample = 1
    epochs = 20
    lr = 0.0002

    check_range_st = 0
    check_range_ed = 129
    pitch_range = check_range_ed - check_range_st-1
    
    device = torch.device('cuda')

    if is_train == 1 :
        train_loader = load_data()
        
        netG = generator(pitch_range).to(device)
        netD = discriminator(pitch_range).to(device)  

        netD.train()
        netG.train()
        
        optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(0.5, 0.999))
        optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(0.5, 0.999)) 
             
        batch_size = 72
        nz = 100
        fixed_noise = torch.randn(batch_size, nz, device=device)
        real_label = 1
        fake_label = 0
        average_lossD = 0
        average_lossG = 0
        average_D_x   = 0
        average_D_G_z = 0

        lossD_list =  []
        lossD_list_all = []
        lossG_list =  []
        lossG_list_all = []
        D_x_list = []
        D_G_z_list = []
        for epoch in range(epochs):
            sum_lossD = 0
            sum_lossG = 0
            sum_D_x   = 0
            sum_D_G_z = 0
            for i, (data,prev_data,chord) in enumerate(train_loader, 0):
                if len(data) == batch_size:
                    ############################
                    # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
                    ###########################
                    # train with real
                    netD.zero_grad()
                    real_cpu = data.to(device)
                    prev_data_cpu = prev_data.to(device)
                    chord_cpu = chord.to(device)

                    batch_size = real_cpu.size(0)
                    label = torch.full((batch_size,), real_label, device=device)
                    D, D_logits, fm = netD(real_cpu,chord_cpu,batch_size,pitch_range)

                    #####loss
                    d_loss_real = reduce_mean(sigmoid_cross_entropy_with_logits(D_logits, 0.9*torch.ones_like(D)))
                    d_loss_real.backward(retain_graph=True)
                    D_x = D.mean().item()
                    sum_D_x += D_x

                    # train with fake
                    noise = torch.randn(batch_size, nz, device=device)
                    fake = netG(noise,prev_data_cpu,chord_cpu,batch_size,pitch_range)
                    label.fill_(fake_label)
                    D_, D_logits_, fm_ = netD(fake.detach(),chord_cpu,batch_size,pitch_range)
                    d_loss_fake = reduce_mean(sigmoid_cross_entropy_with_logits(D_logits_, torch.zeros_like(D_)))
            
                    d_loss_fake.backward(retain_graph=True)
                    D_G_z1 = D_.mean().item()
                    errD = d_loss_real + d_loss_fake
                    errD = errD.item()
                    lossD_list_all.append(errD)
                    sum_lossD += errD
                    optimizerD.step()

                    ############################
                    # (2) Update G network: maximize log(D(G(z)))
                    ###########################
                    netG.zero_grad()
                    label.fill_(real_label)  # fake labels are real for generator cost
                    D_, D_logits_, fm_= netD(fake,chord_cpu,batch_size,pitch_range)

                    ###loss
                    g_loss0 = reduce_mean(sigmoid_cross_entropy_with_logits(D_logits_, torch.ones_like(D_)))
                    #Feature Matching
                    features_from_g = reduce_mean_0(fm_)
                    features_from_i = reduce_mean_0(fm)
                    fm_g_loss1 =torch.mul(l2_loss(features_from_g, features_from_i), 0.1)

                    mean_image_from_g = reduce_mean_0(fake)
                    smean_image_from_i = reduce_mean_0(real_cpu)
                    fm_g_loss2 = torch.mul(l2_loss(mean_image_from_g, smean_image_from_i), 0.01)

                    errG = g_loss0 + fm_g_loss1 + fm_g_loss2
                    errG.backward(retain_graph=True)
                    D_G_z2 = D_.mean().item()
                    optimizerG.step()
                    
                    ############################
                    # (3) Update G network again: maximize log(D(G(z)))
                    ###########################
                    netG.zero_grad()
                    label.fill_(real_label)  # fake labels are real for generator cost
                    D_, D_logits_, fm_ = netD(fake,chord_cpu,batch_size,pitch_range)

                    ###loss
                    g_loss0 = reduce_mean(sigmoid_cross_entropy_with_logits(D_logits_, torch.ones_like(D_)))
                    #Feature Matching
                    features_from_g = reduce_mean_0(fm_)
                    features_from_i = reduce_mean_0(fm)
                    loss_ = nn.MSELoss(reduction='sum')
                    feature_l2_loss = loss_(features_from_g, features_from_i)/2
                    fm_g_loss1 =torch.mul(feature_l2_loss, 0.1)

                    mean_image_from_g = reduce_mean_0(fake)
                    smean_image_from_i = reduce_mean_0(real_cpu)
                    mean_l2_loss = loss_(mean_image_from_g, smean_image_from_i)/2
                    fm_g_loss2 = torch.mul(mean_l2_loss, 0.01)
                    errG = g_loss0 + fm_g_loss1 + fm_g_loss2
                    sum_lossG +=errG
                    errG.backward()
                    lossG_list_all.append(errG.item())

                    D_G_z2 = D_.mean().item()
                    sum_D_G_z += D_G_z2
                    optimizerG.step()
            
                    if epoch % 20 == 0:
                        print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f'
                            % (epoch, epochs, i, len(train_loader),
                                errD, errG, D_x, D_G_z1, D_G_z2))

            if (epoch+1) % 5 == 0:
                vutils.save_image(real_cpu,
                        "file/real_samples.png",
                        normalize=True)
                fake = netG(fixed_noise,prev_data_cpu,chord_cpu,batch_size,pitch_range)
                vutils.save_image(fake.detach(),
                        "file/fake_samples_epoch_{}.png".format(epoch),
                        normalize=True)

                # Checkpoint every 5 epochs
                torch.save(netG.state_dict(), "../models/netG_{}_epoch{}_augmented.pth".format(MODEL_NAME,epoch+1))
                torch.save(netD.state_dict(), "../models/netD_{}_epoch{}_augmented.pth".format(MODEL_NAME,epoch+1))

            average_lossD = (sum_lossD / len(train_loader.dataset))
            average_lossG = (sum_lossG / len(train_loader.dataset))
            average_D_x = (sum_D_x / len(train_loader.dataset))
            average_D_G_z = (sum_D_G_z / len(train_loader.dataset))

            lossD_list.append(average_lossD)
            lossG_list.append(average_lossG)            
            D_x_list.append(average_D_x)
            D_G_z_list.append(average_D_G_z)

            print('==> Epoch: {} Average lossD: {:.10f} average_lossG: {:.10f},average D(x): {:.10f},average D(G(z)): {:.10f} '.format(
              epoch, average_lossD,average_lossG,average_D_x, average_D_G_z)) 

        np.save("file/lossD_{}_list.npy".format(MODEL_NAME),lossD_list)
        np.save("file/lossG_{}_list.npy".format(MODEL_NAME),lossG_list)
        np.save("file/lossD_{}_list_all.npy".format(MODEL_NAME),lossD_list_all)
        np.save("file/lossG_{}_list_all.npy".format(MODEL_NAME),lossG_list_all)
        np.save("file/D_x_{}_list.npy".format(MODEL_NAME),D_x_list)
        np.save("file/D_G_z_{}_list.npy".format(MODEL_NAME),D_G_z_list)

    if is_draw == 1:
        lossD_print = np.load('lossD_list.npy')
        lossG_print = np.load('lossG_list.npy')
        length = lossG_print.shape[0]

        x = np.linspace(0, length-1, length)
        x = np.asarray(x)
        plt.figure()
        plt.plot(x, lossD_print,label=' lossD',linewidth=1.5)
        plt.plot(x, lossG_print,label=' lossG',linewidth=1.5)

        plt.legend(loc='upper right')
        plt.xlabel('data')
        plt.ylabel('loss')
        plt.savefig('where you want to save/lr='+ str(lr) +'_epoch='+str(epochs)+'.png')

    if is_sample == 1:
        batch_size = 8
        nz = 100
        n_bars = 7
        X_te = np.load('X_te_augmented.npy')
        prev_X_te = np.load('prev_X_te_augmented.npy')
        prev_X_te = prev_X_te[:,:,check_range_st:check_range_ed,:]
        y_te    = np.load('chord_te_augmented.npy')
       
        test_iter = get_dataloader(X_te,prev_X_te,y_te)
        
        test_loader = DataLoader(test_iter, batch_size=batch_size, shuffle=False)

        netG = sample_generator(pitch_range)
        netG.load_state_dict(torch.load("../models/netG_{}_epoch19_augmented.pth".format(MODEL_NAME)))

        netG.eval()

        output_songs = []
        output_chords = []
        for i, (data,prev_data,chord) in enumerate(test_loader, 0):
            list_song = []
            first_bar = data[0].view(1,1,16,128)
            list_song.append(first_bar)

            list_chord = []
            first_chord = chord[0].view(1,13).numpy()
            list_chord.append(first_chord)
            noise = torch.randn(batch_size, nz)

            for bar in range(n_bars):
                z = noise[bar].view(1,nz)
                y = chord[bar].view(1,13)
                if bar == 0:
                    prev = data[0].view(1,1,16,128)
                else:
                    prev = list_song[bar-1].view(1,1,16,128)
                sample = netG(z, prev, y, 1,pitch_range)
                list_song.append(sample)
                list_chord.append(y.numpy())

            output_songs.append(list_song)
            output_chords.append(list_chord)
        np.save("output_{}_songs_augmented_19.npy".format(MODEL_NAME),np.asarray(output_songs))
        np.save("output_{}_chords_augmented_19.npy".format(MODEL_NAME),np.asarray(output_chords))

        print('creation completed, check out what I make!')


if __name__ == "__main__" :

    main()

