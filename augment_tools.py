import kornia
import torch
import random
import numpy as np


def augment(im, aug_type):
    if aug_type == 'normal':
        im = im
    # elif aug_type == 'rotated':
    #     rotAng = np.random.choice([-10, 10])
    #     im = korniaAffine(im, rotAng, 'rotate')
    elif aug_type == 'gaussNoise':
        im = augment_gaussian_noise(im, (0.15, 0.3))
    elif aug_type == 'mirror':
        im = torch.flip(im, [-1])
    elif aug_type in ['blur', 'sharpen', 'translate', 'rotate']:
        im = im.unsqueeze(0)
        if aug_type == 'blur':
            im = kornia.filters.gaussian_blur2d(im, (7, 7), (3, 3))
        elif aug_type == 'sharpen':
            im_blur = kornia.filters.gaussian_blur2d(im, (17, 17), (11, 11))
            difference = im - im_blur
            im = im + difference
        elif aug_type == 'translate':
            motion_x = np.random.choice([-20, -10, 10, 20])
            motion_y = np.random.choice([-20, -10, 10, 20])
            translation = torch.Tensor(np.array([[motion_x,
                                                  motion_y]])).cuda()
            im = kornia.geometry.transform.translate(im, translation)
        elif aug_type == 'rotate':
            rotate_angle = np.random.choice([-10, 10])
            center_x, center_y = im.shape[1]//2, im.shape[2]//2
            center = torch.Tensor(np.array[[center_x, center_y]]).cuda()
            im = kornia.geometry.transform.rotate(im, rotate_angle, center)
        im = im[0]
    return im


def korniaAffine(im, parameter, aug_type, dataType='data'):
    '''
    Get rotation by given angle or scale by given factor along axis-0
    using kornia.
    (See https://kornia.readthedocs.io/en/latest/geometry.transform.html)
    '''
    center = torch.ones(1, 2).cuda()
    center[..., 0] = im.shape[1] // 2
    center[..., 1] = im.shape[2] // 2
    if aug_type == 'rotate':
        scale = torch.ones(1).cuda()
        angle = parameter*scale
    elif aug_type == 'scale':
        scale = torch.Tensor([parameter]).cuda()
        angle = 0*scale
        # vol_warped = kornia.scale(vol[:,0,:,:,:],scale,center)
    if dataType == 'data':
        interpolation = 'bilinear'
    elif dataType == 'label':
        interpolation = 'nearest'
    M = kornia.get_rotation_matrix2d(center, angle, scale)
    _, h, w = im.shape
    im_warped = kornia.warp_affine(im[None, :, :, :].float(), M.cuda(),
                                   dsize=(h, w), flags=interpolation)
    # vol_warped = vol_warped[:,None,:,:,:]
    return im_warped[0]


def korniaTranslate(im, choice, dataType='data'):
    '''
    Random translation using kornia translate transform.
    Additional function needed for construction of translation Tensor.
    Choice has format str(magnitude)+'axis' where axis = x or y.
    '''
    axis = choice[-1]
    transMag = int(choice[:-1])
    transVal = torch.zeros(im.shape[0], 2)
    if axis == 'x':
        transVal[:, 0] = transMag
    elif axis == 'y':
        transVal[:, 1] = transMag
    if dataType == 'data':
        interpolation = 'bilinear'
    elif dataType == 'label':
        interpolation = 'nearest'
    M = kornia.geometry.transform.affwarp._compute_translation_matrix(transVal)
    _, _, h, w = im.shape
    vol_warped = kornia.warp_affine(im.float(), M.cuda(), dsize=(h, w),
                                    flags=interpolation)
    return vol_warped


def augment_gaussian_noise(data_sample, noise_variance=(0, 0.1)):
    '''
    Modified function from batchgenerators to process cuda tensor.
    '''
    if noise_variance[0] == noise_variance[1]:
        variance = noise_variance[0]
    else:
        variance = random.uniform(noise_variance[0], noise_variance[1])
    data_sample = data_sample + torch.Tensor(
        np.random.normal(0.0, variance, size=data_sample.shape)).cuda()
    return data_sample
