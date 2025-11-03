import torch
from torch.utils.data import DataLoader, random_split
from Sonarpcd_data import SonarPCDData
from sonar_pcdnet import SonarPCDNet
from utils import get_dataloader_workers, sinkhorn_emd, rotvec2mat, rotation_matrix_loss
import torch.nn as nn
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from torch.nn.utils import clip_grad

if __name__ == '__main__':
    wholedata = SonarPCDData('/media/kemove/043E0D933E0D7F44/Sonar_Diffusion_SLAM/data')
    train_size = int(0.8*len(wholedata))
    test_size = len(wholedata)-train_size
    traindata, testdata = random_split(wholedata, [train_size, test_size])

    train_dataloader = DataLoader(
        traindata,
        batch_size=1,
        shuffle=True,
        num_workers=get_dataloader_workers()
    )
    test_dataloader = DataLoader(
        testdata,
        batch_size=1,
        shuffle=True,
        num_workers=get_dataloader_workers()
    )
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SonarPCDNet().to(device)
    # === 冻结 BatchNorm 统计 ===
    def freeze_bn(m):
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
            m.eval()
    model.apply(freeze_bn)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-6)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.9)
    mseloss = nn.MSELoss()
    summary_writer = SummaryWriter('/media/kemove/043E0D933E0D7F44/Sonar_Diffusion_SLAM/trainlog/exp4')
    Epochs = 100

    for epoch in range(Epochs):
        print(f'epoch: {epoch}')
        
        print('start training:')
        rloss_train = 0.0
        tloss_train = 0.0
        # emd_train = 0.0
        # x_train = 0.0
        # y_train = 0.0
        # z_train = 0.0
        pcd_train = 0.0
        entropy_train = 0.0
        model.train()
        model.apply(freeze_bn)
        for i, (pcd1, pcd2, f1, f2, R, t, pts1_gt, pts2_gt) in tqdm(enumerate(train_dataloader), total=train_size):
            pcd1 = pcd1.to(device).float()
            pcd2 = pcd2.to(device).float()
            f1 = f1.to(device).float()
            f2 = f2.to(device).float()

            R = R.to(device).float()
            t = t.to(device).float()
            pts1_gt = pts1_gt.to(device).float()
            pts2_gt = pts2_gt.to(device).float()
            # rs = torch.norm(pts1_gt, dim=2)
            # thetas = torch.atan2(pts1_gt[:, :, 1], pts1_gt[:, :, 0])
            # rtheta torch.cat((rs.unsqueeze(2), thetas.unsqueeze(2)), dim=2) #

            pts1_pre = model(pcd1, f1, pcd2, f2, pts1_gt.shape[1], f1[:, ::10, :2])
            pts2_pre = model(pcd2, f2, pcd1, f1, pts2_gt.shape[1], f2[:, ::10, :2])

            q1 = pts1_pre-torch.mean(pts1_pre, dim=1)
            q2 = pts2_pre-torch.mean(pts2_pre, dim=1)
            # q2 = pts2_gt-torch.mean(pts2_gt, dim=1)
            W = torch.einsum('bni,bnj->bij', q2, q1).squeeze(0)
            U, S, V = torch.linalg.svd(W)

            # R_pre = (U @ V).t()
            det = torch.linalg.det((U @ V).t())
            D = torch.eye(3, device=device)
            D[2, 2] = torch.sign(det)
            R_pre = V.t() @ D @ U.t()
                
            t_pre = torch.mean(pts1_pre, dim=1).squeeze(0)-R_pre @ torch.mean(pts2_pre, dim=1).squeeze(0)
            pcd1_loss = torch.mean(torch.norm(pts1_pre-pts1_gt, dim=2))#mseloss(pts1_pre, pts1_gt)#
            pcd2_loss = torch.mean(torch.norm(pts2_pre-pts2_gt, dim=2))
            # x_loss = mseloss(pts1_pre[:, :, 0], pts1_gt[:, :, 0])
            # y_loss = mseloss(pts1_pre[:, :, 1], pts1_gt[:, :, 1])
            # z1_loss = mseloss(pts1_pre[:, :, 2], pts1_gt[:, :, 2])
            # z2_loss = mseloss(pts2_pre[:, :, 2], pts2_gt[:, :, 2])
            # entropy_loss = (-torch.sum(weights1*torch.log(weights1+1e-10), dim=-1)).mean()#+(-torch.sum(weights2*torch.log(weights2+1e-10), dim=-1)).mean()

            t_loss = torch.linalg.norm(t.squeeze(0)-t_pre)
            R_loss = torch.abs(R.squeeze(0)-R_pre).sum()
            # cd = chamfer_distance(pts_pre, pts_gt)
            # emd12 = sinkhorn_emd(pts1_gt, pts1_pre, eps=1e-3)
            loss = R_loss+t_loss+pcd1_loss+pcd2_loss #+entropy_loss*1e-6

            rloss_train += R_loss
            tloss_train += t_loss
            # emd_train += emd12#+emd21
            # x_train += x_loss
            # y_train += y_loss
            # z_train += z1_loss
            # entropy_train += entropy_loss
            pcd_train += 0.5*(pcd1_loss+pcd2_loss)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # for name, param in model.named_parameters():
            #     if param.grad is not None:
            #         print(name, param.grad.abs().mean().item())
        scheduler.step()
        torch.cuda.empty_cache()

        rloss_train = rloss_train/train_size
        tloss_train = tloss_train/train_size
        # emd_train = emd_train/train_size
        # x_train = x_train/train_size
        # y_train = y_train/train_size
        # z_train = z_train/train_size
        # entropy_train = entropy_train/train_size
        pcd_train = pcd_train/train_size

        summary_writer.add_scalar('train/rloss', rloss_train, epoch)
        summary_writer.add_scalar('train/tloss', tloss_train, epoch)
        # summary_writer.add_scalar('train/emd', emd_train, epoch)
        # summary_writer.add_scalar('train/x', x_train, epoch)
        # summary_writer.add_scalar('train/y', y_train, epoch)
        # summary_writer.add_scalar('train/z', z_train, epoch)
        # summary_writer.add_scalar('train/entropy', entropy_train, epoch)
        summary_writer.add_scalar('train/pcdloss', pcd_train, epoch)
        
        print('start testing:')
        rloss_test = 0.0
        tloss_test = 0.0
        # emd_test = 0.0
        # x_test = 0.0
        # y_test = 0.0
        # z_test = 0.0
        # entropy_test = 0.0
        pcd_test = 0.0
        model.eval()
        for j, (pcd1, pcd2, f1, f2, R, t, pts1_gt, pts2_gt) in tqdm(enumerate(test_dataloader), total=test_size):
            with torch.no_grad():
                pcd1 = pcd1.to(device).float()
                pcd2 = pcd2.to(device).float()
                f1 = f1.to(device).float()
                f2 = f2.to(device).float()

                R = R.to(device).float()
                t = t.to(device).float()
                pts1_gt = pts1_gt.to(device).float()
                pts2_gt = pts2_gt.to(device).float()

                pts1_pre = model(pcd1, f1, pcd2, f2, pts1_gt.shape[1], f1[:, ::10, :2])
                pts2_pre = model(pcd2, f2, pcd1, f1, pts2_gt.shape[1], f2[:, ::10, :2])

                q1 = pts1_pre-torch.mean(pts1_pre, dim=1)
                q2 = pts2_pre-torch.mean(pts2_pre, dim=1)
                # q2 = pts2_gt-torch.mean(pts2_gt, dim=1)
                W = torch.einsum('bni,bnj->bij', q2, q1).squeeze(0)
                U, S, V = torch.linalg.svd(W)

                # R_pre = (U @ V).t()
                det = torch.linalg.det((U @ V).t())
                D = torch.eye(3, device=device)
                D[2, 2] = torch.sign(det)
                R_pre = V.t() @ D @ U.t()
                    
                t_pre = torch.mean(pts1_pre, dim=1).squeeze(0)-R_pre @ torch.mean(pts2_pre, dim=1).squeeze(0)
                pcd1_loss = torch.mean(torch.norm(pts1_pre-pts1_gt, dim=2))#mseloss(pts1_pre, pts1_gt)#
                pcd2_loss = torch.mean(torch.norm(pts2_pre-pts2_gt, dim=2))
                # x_loss = mseloss(pts1_pre[:, :, 0], pts1_gt[:, :, 0])
                # y_loss = mseloss(pts1_pre[:, :, 1], pts1_gt[:, :, 1])
                # z1_loss = mseloss(pts1_pre[:, :, 2], pts1_gt[:, :, 2])
                # z2_loss = mseloss(pts2_pre[:, :, 2], pts2_gt[:, :, 2])
                # entropy_loss = (-torch.sum(weights1*torch.log(weights1+1e-10), dim=-1)).mean()#+(-torch.sum(weights2*torch.log(weights2+1e-10), dim=-1)).mean()

                t_loss = torch.linalg.norm(t.squeeze(0)-t_pre)
                R_loss = torch.abs(R.squeeze(0)-R_pre).sum()
        
                # cd = chamfer_distance(pts_pre, pts_gt)
                # emd12 = sinkhorn_emd(pts1_gt, pts1_pre, eps=1e-3)
                rloss_test += R_loss
                tloss_test += t_loss
                pcd_test += 0.5*(pcd1_loss+pcd2_loss)
                # emd_train += emd12#+emd21
                # x_train += x_loss
                # y_train += y_loss
                # z_test += z1_loss
                # entropy_test += entropy_loss

                # emd_test += emd12
                # x_test += x_loss
                # y_test += y_loss
            torch.cuda.empty_cache()
        rloss_test = rloss_test/test_size
        tloss_test = tloss_test/test_size
        # emd_test = emd_test/test_size
        # x_test = x_test/test_size
        # y_test = y_test/test_size
        # z_test = z_test/test_size
        # entropy_test = entropy_test/test_size
        pcd_test = pcd_test/test_size

        summary_writer.add_scalar('test/rloss', rloss_test, epoch)
        summary_writer.add_scalar('test/tloss', tloss_test, epoch)
        # summary_writer.add_scalar('test/emd', emd_test, epoch)
        # summary_writer.add_scalar('test/x', x_test, epoch)
        # summary_writer.add_scalar('test/y', y_test, epoch)
        # summary_writer.add_scalar('test/z', z_test, epoch)
        # summary_writer.add_scalar('test/entropy', entropy_test, epoch)
        summary_writer.add_scalar('test/pcdloss', pcd_test, epoch)

        # if epoch % 10 == 0:
        #     print(f'train loss: {loss_train}, qloss: {qloss_train}, tloss: {tloss_train}')
        #     print(f'test loss: {loss_test}, qloss: {qloss_test}, tloss: {tloss_test}')
        if epoch % 5 == 0 or epoch == Epochs-1:
            checkpoints = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict()
            }
            torch.save(checkpoints, f'/media/kemove/043E0D933E0D7F44/Sonar_Diffusion_SLAM/trainlog/exp4/model_{epoch}.pth')