


import pygame
import numpy as np
import os
from PIL import Image

pygame.init()
# screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN) # 1440, 900
screen = pygame.display.set_mode((512, 512))

width = screen.get_width()
height = screen.get_height()
scale = 1
p = 50
x0 = 0.5
y0 = 0.2
sigma = 0.01

w = width//scale
h = height//scale

dt = 1e-2

'''
safe_dt = (1/(w/scale))**2
print(safe_dt)
assert dt < safe_dt
'''

f0 = lambda x, y: np.exp(-((x - x0)**2 + (y - y0)**2) / (2 * sigma**2)) * np.exp(1j * 2 * np.pi * p * (y - y0))

def normalize(psi,dx,dy):
    psi_pdf = np.abs(psi)**2
    psi_norm = psi_pdf.sum() * dx * dy
    psi = psi / psi_norm
    psi_pdf = psi_pdf / psi_norm
    return psi, psi_pdf

def filter_wall(V,k=0.1):
    # V: (w,h)
    lap = np.roll(V,1,axis=0) + np.roll(V,-1,axis=0) + np.roll(V,1,axis=1) + np.roll(V,-1,axis=1) # - 4*V
    V = V + k * lap
    return V

# d psi / d t = -i * H_bar(psi)
# H_bar(psi) = -laplace(psi) + V * psi
# V = 0
# F (H_bar(psi)) = - (kx**2 + ky**2) * F(psi)
# H_bar(psi) = iF (F H_bar(psi))
# psi = psi + dt (-i * H_bar(psi))

class Psi:
    def __init__(self, width, height, f0, scale=1):
        w = width//scale
        h = height//scale
        self.arr = np.zeros((width,height,3))
        
        self.scale = scale
        self.w = w
        self.h = h
        
        psi = np.zeros((w,h,2))

        x = np.arange(w).astype(np.complex128)/w
        y = np.arange(h).astype(np.complex128)/h
        x, y = np.meshgrid(x, y)

        self.x = x
        self.y = y

        kx = 2*np.pi*np.fft.fftfreq(w)
        ky = 2*np.pi*np.fft.fftfreq(h)
        kx,ky = np.meshgrid(kx,ky) # (Freq_x,Freq_y)
        kr_sq = kx**2 + ky**2
        self.kx = kx
        self.ky = ky
        self.kr_sq = kr_sq

        psi = f0(x,y) # (w,h)
        psi, psi_pdf = normalize(psi, 1/w, 1/h)
        self.psi = psi
        self.psi_pdf = psi_pdf
        self.cal_Exy()
        self.cal_Dxy()
        self.arr_psi = np.repeat(np.repeat(psi, self.scale, axis=0), self.scale, axis=1)
        self.arr_psi_pdf = np.repeat(np.repeat(psi_pdf, self.scale, axis=0), self.scale, axis=1)
    def enforce_boundary(self):
        psi = self.psi
        psi[0,:] = 0
        psi[-1,:] = 0
        psi[:,0] = 0
        psi[:,-1] = 0
        return psi
    def cal_Ep(self,p):
        # p: (w,h)
        psi_pdf = self.psi_pdf
        w,h = self.w, self.h
        x,y = self.x, self.y
        dx,dy = 1/w, 1/h
        ds = dx * dy
        Ep = (psi_pdf * p).sum() * ds
        return Ep
    def cal_Exy(self):
        x,y = self.x, self.y
        Ex = self.cal_Ep(x)
        Ey = self.cal_Ep(y)
        xy = np.array([Ex, Ey]) # (2,)
        self.Exy = xy
    def cal_Dxy(self):
        dx,dy = 1/w, 1/h
        Ex,Ey = self.Exy
        r = (dx-Ex)**2 + (dy-Ey)**2
        Dr = self.cal_Ep(r)
        self.Dr = Dr
    def update(self, V):
        # V: (w,h)
        w,h = self.w, self.h
        kr_sq = self.kr_sq # (w,h)
        scale = self.scale
        psi = self.psi
        
        F_psi = np.fft.fft2(psi)
        F_lap_psi = - kr_sq * F_psi
        F_V_psi = np.fft.fft2(V * psi)

        F_HBar_psi = -F_lap_psi + F_V_psi

        HBar_psi = np.fft.ifft2(F_HBar_psi) # (w,h)
        psi = psi + dt * (- 1j * HBar_psi)

        self.enforce_boundary()
        psi, psi_pdf = normalize(psi, 1/w, 1/h)
        self.psi = psi
        self.psi_pdf = psi_pdf
        self.cal_Exy()
        self.cal_Dxy()
        
        self.arr_psi = np.repeat(np.repeat(psi, self.scale, axis=0), self.scale, axis=1)
        self.arr_psi_pdf = np.repeat(np.repeat(psi_pdf, self.scale, axis=0), self.scale, axis=1)
    def get_posMask_r(self, r):
        Ex,Ey = self.Exy
        x,y = self.x, self.y
        pos_mask = (x-Ex)**2 + (y-Ey)**2 < r**2
        return pos_mask
    def draw(self, screen):
        psi_pdf = self.arr_psi_pdf # (w,h,1)
        w,h = psi_pdf.shape[:2]
        
        sigma = self.Dr ** 0.5
        pos_mask = self.get_posMask_r(sigma)
        
        arr = self.arr
        real = np.real(self.arr_psi)
        imag = np.imag(self.arr_psi)
        
        real_abs = np.abs(real)
        imag_abs = np.abs(imag)
        
        mean_real = real_abs[pos_mask].mean()
        mean_imag = imag_abs[pos_mask].mean()
        
        # print(real.max())
        arr[:w, :h, 0] = (real) * 200 / mean_real
        arr[:w, :h, 2] = (imag) * 200 / mean_imag
        arr = np.clip(arr, 0, 255)
        
        self.img_arr = arr[::self.scale,::self.scale]
        
        pygame.surfarray.blit_array(screen, arr)



psi = Psi(width,height,f0,scale)
clock = pygame.time.Clock()

mask = np.zeros((width,height))

# mask[(height//2-5):(height//2+5),:] = 1
# mask[:,(width//2-5):(width//2+5)] = 0

mask_disp = mask * 255
mask_disp = mask_disp.T
mask_disp = np.stack([mask_disp]*4, axis=-1).astype(np.uint8)
mask_disp = pygame.image.frombuffer(mask_disp.astype(np.uint8).tobytes(), (width, height), 'RGBA')


scale = psi.scale

mask = mask[scale//2:-(scale//2)+1:scale,scale//2:-(scale//2)+1:scale]


x,y = psi.x, psi.y
V = -0.0000 / ((x-0.5)**2 + (y-0.5)**2 + 1e-3)

# 对V进行许多次滤波
for i in range(10):
    V = filter_wall(V)

save_dir = os.makedirs(os.path.join("field_state","electron_img","inv_x_square"), exist_ok=True)

abs_time = 0
while True:
    for e in pygame.event.get():
        if e.type == pygame.QUIT:
            pygame.quit()
        
    screen.fill((0,0,0))
    psi.update(V)
    psi.draw(screen)
    screen.blit(mask_disp, (0, 0))
    pygame.display.flip()
    clock.tick(60)
    # 保存图像
    Image.fromarray(psi.img_arr.astype(np.uint8)).save(os.path.join(save_dir,f"frame_{abs_time:04d}.jpg")\
        , quality=80)
    abs_time += 1


'''
# save psi.npy (保存psi.psi)
save_dict = {'psi': psi.psi, "V": V}
np.save(os.path.join('field_state','electron_state','psi.npy'), save_dict)
'''


# ffmpeg -framerate 1000 -i frame_%d.png -c:v libx264 -preset veryfast -crf 20 out.mp4
