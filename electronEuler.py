# 台球模拟
import pygame
import numpy as np
import os
from PIL import Image

def cal_laplace(psi):
    # psi: (w,h,2)
    lap = np.roll(psi,1,axis=0) + np.roll(psi,-1,axis=0) + np.roll(psi,1,axis=1) + np.roll(psi,-1,axis=1) - 4*psi
    return lap


pygame.init()
# screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN) # 1440, 900
screen = pygame.display.set_mode((512, 512))

w = screen.get_width()
h = screen.get_height()

scale = 20 # scale = 7 时能看到复杂图像，但是演化非常慢
sigma = 0.1
dt = 1e-3
p = 10

x0 = 0.5
y0 = 0.2


f0x = lambda x, y: np.exp(-((x - x0)**2 + (y - y0)**2) / (2 * sigma**2)) * np.cos(2 * np.pi * p * (y - y0))
f0y = lambda x, y: np.exp(-((x - x0)**2 + (y - y0)**2) / (2 * sigma**2)) * np.sin(2 * np.pi * p * (y - y0))




safe_dt = (1/(w/scale))**2
print(safe_dt)
assert dt < safe_dt

class Psi:
    def __init__(self, w, h, f0x, f0y, scale=1):
        w_ori = w
        h_ori = h
        w = w//scale
        h = h//scale
        self.w = w
        self.h = h
        self.scale = scale
        self.w_ori = w_ori
        self.h_ori = h_ori
        self.arr = np.zeros((w_ori,h_ori,3))
        
        psi = np.zeros((w,h,2))

        x = np.arange(w)/w
        y = np.arange(h)/h
        x, y = np.meshgrid(x, y)
        self.x = x
        self.y = y


        psi[:,:,0] = f0x(x,y)
        psi[:,:,1] = f0y(x,y)
        psi_norm = np.linalg.norm(psi, axis=-1, keepdims=True).sum()
        psi = psi / (psi_norm) / (1/w) / (1/h)
        psi_pdf = np.linalg.norm(psi, axis=-1)
        self.psi = psi
        self.psi_pdf = psi_pdf
    def H_bar(self, V, psi):
        psi_lap = cal_laplace(psi)
        return -psi_lap + V * psi
    def update(self, V):
        # V: (w,h)
        V = V[:,:,None] # (w,h,1)

        w,h = self.w, self.h
        psi = self.psi
        
        H1_psi = self.H_bar(V, psi) # ()
        H2_psi = self.H_bar(V, H1_psi)
        
        iH1_psi = np.stack([-H1_psi[:,:,1], H1_psi[:,:,0]], axis=-1)
        
        psi = psi - iH1_psi * dt - 0.5 * H2_psi * dt**2

        psi_norm = np.linalg.norm(psi, axis=-1, keepdims=True).sum()
        psi = psi / (psi_norm) / (1/w) / (1/h)
        psi_pdf = np.linalg.norm(psi, axis=-1) # (w,h)
        self.psi = psi
        self.psi_pdf = psi_pdf
        
        self.arr_psi = np.repeat(np.repeat(psi, self.scale, axis=0), self.scale, axis=1)
        self.arr_psi_pdf = np.repeat(np.repeat(psi_pdf, self.scale, axis=0), self.scale, axis=1)
        
    def draw(self, screen):
        psi_pdf = self.arr_psi_pdf # (w,h,1)
        w,h = psi_pdf.shape[:2]
        
        arr = self.arr
        real = self.arr_psi[:,:,0]
        imag = self.arr_psi[:,:,1]
        # print(real.max())
        arr[:w, :h, 0] = np.abs(real) * 100
        arr[:w, :h, 2] = np.abs(imag) * 100
        arr = np.clip(arr, 0, 255)
        
        self.img_arr = arr[::self.scale,::self.scale]
        
        pygame.surfarray.blit_array(screen, arr)


psi = Psi(w,h,f0x,f0y,scale)
clock = pygame.time.Clock()

mask = np.zeros((w,h))

# mask[:,(h//2-5):(h//2+5)] = 1

mask_disp = mask * 255
mask_disp = np.stack([mask_disp]*4, axis=-1).astype(np.uint8)
mask_disp = pygame.image.frombuffer(mask_disp.astype(np.uint8).tobytes(), (w, h), 'RGBA')


scale = psi.scale

mask = mask[scale//2:-(scale//2)+1:scale,scale//2:-(scale//2)+1:scale]


x,y = psi.x, psi.y
V = mask * 1000

os.makedirs(os.path.join("field_state","electron_img_euler"), exist_ok=True)

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
    Image.fromarray(psi.img_arr.astype(np.uint8)).save(os.path.join("field_state","electron_img_euler",f"frame_{abs_time:04d}.jpg")\
        , quality=80)
    abs_time += 1


'''
# save psi.npy (保存psi.psi)
save_dict = {'psi': psi.psi, "V": V}
np.save(os.path.join('field_state','electron_state','psi.npy'), save_dict)
'''


# ffmpeg -framerate 1000 -i frame_%d.png -c:v libx264 -preset veryfast -crf 20 out.mp4
