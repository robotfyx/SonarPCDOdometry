import torch
from torch.utils.data import DataLoader, random_split
from Sonarpcd_data import SonarPCDData
from sonar_pcdnet import SonarPCDNet
from utils import get_dataloader_workers, pad, sinkhorn_emd, rotvec2mat, rotation_matrix_loss
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

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.95)
    mseloss = nn.MSELoss()
    summary_writer = SummaryWriter('/media/kemove/043E0D933E0D7F44/Sonar_Diffusion_SLAM/trainlog/exp2')
    Epochs = 100

    for epoch in range(Epochs):
        print(f'epoch: {epoch}')
        
        print('start training:')
        rvloss_train = 0.0
        tloss_train = 0.0
        emd_train = 0.0
        pcd_train = 0.0
        # cyc_train = 0.0
        # var_train = 0.0
        model.train()
        model.apply(freeze_bn)
        for i, (pcd1, pcd2, f1, f2, rv, t, pts1_gt, pts2_gt) in tqdm(enumerate(train_dataloader), total=train_size):
            pcd1 = pcd1.to(device).float()
            pcd2 = pcd2.to(device).float()
            f1 = f1.to(device).float()
            f2 = f2.to(device).float()
            if pcd1.shape[1] < 2048:
                pcd1 = pad(pcd1)
                pcd2 = pad(pcd2)
                f1 = pad(f1)
                f2 = pad(f2)

            rv = rv.to(device).float()
            t = t.to(device).float()
            pts1_gt = pts1_gt.to(device).float()
            pts2_gt = pts2_gt.to(device).float()

            pts1_pre = model(pcd1, f1, pcd2, f2, pts1_gt.shape[1])
            pcd_loss = torch.mean(torch.norm(pts1_pre-pts1_gt, dim=2))
            
            # # 1->2
            # rv12_pre, t12_pre, pts1_pre = model(pcd1, f1, pcd2, f2)
            # # r12pre = rotvec2mat(rv12_pre)
        
            # rv12_loss = mseloss(rv12_pre, rv)#torch.norm(rv12_pre-rv, dim=1).mean() #rotation_matrix_loss(r12pre, rgt)#
            # t12_loss = mseloss(t12_pre, t) #torch.norm(t12_pre-t, dim=1).mean()
            # cd = chamfer_distance(pts_pre, pts_gt)
            emd12 = sinkhorn_emd(pts1_gt, pts1_pre)
            '''
            # 2->1
            rv21_pre, t21_pre, pts2_pre = model(pcd2, f2, pcd1, f1)
            emd21 = sinkhorn_emd(pts2_gt, pts2_pre)
            # cycle loss            
            # R12 = euler2mat(rv12_pre, deg=True)
            T12 = torch.eye(4, device=device, dtype=torch.float32)
            T12[:3, :3] = r12pre
            T12[:3, 3] = t12_pre.squeeze(0)
            # R21 = euler2mat(rv21_pre, deg=True)
            T21 = torch.eye(4, device=device, dtype=torch.float32)
            T21[:3, :3] = rotvec2mat(rv21_pre)
            T21[:3, 3] = t21_pre.squeeze(0)
            cyc_loss = torch.sum(torch.abs(T12 @ T21-torch.eye(4, device=device, dtype=torch.float32)))
            '''
            loss = pcd_loss+emd12

            # rvloss_train += rv12_loss
            # tloss_train += t12_loss
            emd_train += emd12#+emd21
            pcd_train += pcd_loss
            # cyc_train += cyc_loss
            # var_train += var_loss*0.5

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # for name, param in model.named_parameters():
            #     if param.grad is not None:
            #         print(name, param.grad.abs().mean().item())
        scheduler.step()
        torch.cuda.empty_cache()

        # rvloss_train = rvloss_train/train_size
        # tloss_train = tloss_train/train_size
        emd_train = emd_train/train_size
        pcd_train = pcd_train/train_size
        # cyc_train = cyc_train/train_size
        # var_train = var_train/train_size
        # summary_writer.add_scalar('train/rvloss', rvloss_train, epoch)
        # summary_writer.add_scalar('train/tloss', tloss_train, epoch)
        summary_writer.add_scalar('train/emd', emd_train, epoch)
        summary_writer.add_scalar('train/pcd', pcd_train, epoch)
        # summary_writer.add_scalar('train/cyc', cyc_train, epoch)
        # summary_writer.add_scalar('train/var', var_train, epoch)
        
        print('start testing:')
        # rvloss_test = 0.0
        # tloss_test = 0.0
        emd_test = 0.0
        pcd_test = 0.0
        # cyc_test = 0.0
        # var_test = 0.0
        model.eval()
        for j, (pcd1, pcd2, f1, f2, rv, t, pts1_gt, pts2_gt) in tqdm(enumerate(test_dataloader), total=test_size):
            with torch.no_grad():
                pcd1 = pcd1.to(device).float()
                pcd2 = pcd2.to(device).float()
                f1 = f1.to(device).float()
                f2 = f2.to(device).float()
                if pcd1.shape[1] < 2048:
                    pcd1 = pad(pcd1)
                    pcd2 = pad(pcd2)
                    f1 = pad(f1)
                    f2 = pad(f2)

                rv = rv.to(device).float()
                t = t.to(device).float()
                pts1_gt = pts1_gt.to(device).float()
                pts2_gt = pts2_gt.to(device).float()

                pts1_pre = model(pcd1, f1, pcd2, f2, pts1_gt.shape[1])
                pcd_loss = torch.mean(torch.norm(pts1_pre-pts1_gt, dim=2))
                
                # # 1->2
                # rv12_pre, t12_pre, pts1_pre = model(pcd1, f1, pcd2, f2)
                # r12pre = rotvec2mat(rv12_pre)

                # rv12_loss = torch.norm(rv12_pre-rv, dim=1).mean()#rotation_matrix_loss(r12pre, rgt)#torch.norm(rv12_pre-rv, dim=1)
                # t12_loss = torch.norm(t12_pre-t, dim=1).mean()
                # cd = chamfer_distance(pts_pre, pts_gt)
                emd12 = sinkhorn_emd(pts1_gt, pts1_pre)
                
                # 2->1
                # rv21_pre, t21_pre, pts2_pre = model(pcd2, f2, pcd1, f1)
                # emd21 = sinkhorn_emd(pts2_gt, pts2_pre)
                # cycle loss            
                # R12 = euler2mat(rv12_pre, deg=True)
                # T12 = torch.eye(4, device=device, dtype=torch.float32)
                # T12[:3, :3] = r12pre
                # T12[:3, 3] = t12_pre.squeeze(0)
                # # R21 = euler2mat(rv21_pre, deg=True)
                # T21 = torch.eye(4, device=device, dtype=torch.float32)
                # T21[:3, :3] = rotvec2mat(rv21_pre)
                # T21[:3, 3] = t21_pre.squeeze(0)
                # cyc_loss = torch.sum(torch.abs(T12 @ T21-torch.eye(4, device=device, dtype=torch.float32)))

                # rvloss_test += rv12_loss
                # tloss_test += t12_loss
                emd_test += emd12
                pcd_test += pcd_loss
                # cyc_test += cyc_loss
            torch.cuda.empty_cache()
        # rvloss_test = rvloss_test/test_size
        # tloss_test = tloss_test/test_size
        emd_test = emd_test/test_size
        pcd_test = pcd_test/test_size
        # cyc_test = cyc_test/test_size
        # var_test = var_test/test_size
        # summary_writer.add_scalar('test/rvloss', rvloss_test, epoch)
        # summary_writer.add_scalar('test/tloss', tloss_test, epoch)
        summary_writer.add_scalar('test/emd', emd_test, epoch)
        summary_writer.add_scalar('test/pcd', pcd_test, epoch)
        # summary_writer.add_scalar('test/cyc', cyc_test, epoch)
        # summary_writer.add_scalar('test/var', var_test, epoch)

        # if epoch % 10 == 0:
        #     print(f'train loss: {loss_train}, qloss: {qloss_train}, tloss: {tloss_train}')
        #     print(f'test loss: {loss_test}, qloss: {qloss_test}, tloss: {tloss_test}')
        if epoch % 10 == 0 or epoch == Epochs-1:
            checkpoints = {
                "model": model.state_dict()
            }
            torch.save(checkpoints, f'/media/kemove/043E0D933E0D7F44/Sonar_Diffusion_SLAM/trainlog/exp2/model_{epoch}.pth')