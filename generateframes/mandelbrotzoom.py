import torch
import numpy as np
import torchvision

torch.set_printoptions(edgeitems=6, linewidth=200)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# FRAMES_PER_SECOND = 24
FRAMES_PER_SECOND = 12
TOTAL_ITERATION = 350

# ComplexPlane: iterates towards one resolution of fractal set
class ComplexPlane:
    def __init__(self,center,real,imag,px_x,px_y):
        self.real_center = center[0]
        self.imag_center = center[1]
        # Positive and negative extent of real/imaginary axis
        self.real_p = real[0]
        self.real_n = real[1]
        self.imag_p = imag[0]
        self.imag_n = imag[1]
        # Image size
        self.px_x = px_x
        self.px_y = px_y
        # Complex Plane
        self.plane = torch.tensor([
            [np.linspace(self.real_center-self.real_n,self.real_center+self.real_p,num=px_x)
                for i in np.arange(px_y)],
            [np.repeat(i,px_x)
                for i in np.linspace(self.imag_center+self.imag_p,self.imag_center-self.imag_n,num=px_y)]
        ], dtype=torch.float32).to(device)
        # Store angle of rotation for generating new plane when zooming
        self.angle = 0
        # Value of iteration starting at z0 = 0+0i
        self.iteration = torch.tensor((), dtype=torch.float32).new_zeros((2,px_y,px_x)).to(device)
        # Counting iterations
        self.canvas = torch.tensor((), dtype=torch.float32).new_zeros((1,px_y,px_x)).to(device)
        pass

    # GENERATING FRACTAL
    # Apply polynomial once
    def polynomial(self,power):
        # Compute new iteration
        next_iteration = self.iteration.clone()
        # Complex multiplication for polynomials
        while power > 1:
            power -= 1
            # Complex multiplication: (a+bi)(c+di) = (ac-bd) + (bc+ad)i
            # Doesn't seem faster than Karatsuba multiplication
            z1_real = next_iteration[0,:,:].clone() #a
            z1_imag = next_iteration[1,:,:].clone() #b
            z2_real = self.iteration[0,:,:].clone() #c
            z2_imag = self.iteration[1,:,:].clone() #d
            next_iteration[0,:,:] = z1_real*z2_real - z1_imag*z2_imag
            next_iteration[1,:,:] = z1_imag*z2_real + z1_real*z2_imag
            del z1_real, z1_imag, z2_real, z2_imag
        #add constant in polynomial (from plane)
        next_iteration[0,:,:] += self.plane[0,:,:].clone()
        next_iteration[1,:,:] += self.plane[1,:,:].clone()
        # print(self.iteration)
        self.iteration = next_iteration
        del next_iteration
        pass

    # # Anti-aliasing of mandelbrot set
    # def supersample_polynomial(self,power,pix,pix_distance,prev_iter):
    #     # Generate super sample complex plane
    #     super_factor = 4
    #     plane = torch.tensor([
    #         [np.linspace(pix[0]-pix_distance[0],pix[0]+pix_distance[0],num=super_factor)
    #             for i in np.arange(super_factor)],
    #         [np.repeat(i,super_factor)
    #             for i in np.linspace(pix[1]+pix_distance[1],pix[1]-pix_distance[1],num=super_factor)]
    #     ], dtype=torch.float32).to(device)
    #     # Apply polynomial
    #     next_iteration = torch.tensor((), dtype=torch.float32).new_zeros(plane.size()).to(device)
    #     next_iteration[0,:,:] = prev_iter[0]
    #     next_iteration[1,:,:] = prev_iter[1]
    #     og_iteration = next_iteration.clone()
    #     while power > 1:
    #         power -= 1
    #         #complex multiplication for power
    #         #complex multiplication: (a+bi)(c+di) = (ac-bd) + (bc+ad)i
    #         z1_real = next_iteration[0,:,:].clone() #a
    #         z1_imag = next_iteration[1,:,:].clone() #b
    #         z2_real = og_iteration[0,:,:].clone() #c
    #         z2_imag = og_iteration[1,:,:].clone() #d
    #         next_iteration[0,:,:] = z1_real*z2_real - z1_imag*z2_imag
    #         next_iteration[1,:,:] = z1_imag*z2_real + z1_real*z2_imag
    #         del z1_real, z1_imag, z2_real, z2_imag
    #     #add constant in polynomial (from plane)
    #     next_iteration[0,:,:] += plane[0,:,:].clone()
    #     next_iteration[1,:,:] += plane[1,:,:].clone()
    #     # print(self.iteration)
    #     supersample = torch.tensor([torch.mean(next_iteration[0,:,:]),torch.mean(next_iteration[1,:,:])]).to(device)
    #     del plane, next_iteration, og_iteration
    #     return supersample

    # Count escape time for each value in complex plane
    def count_iteration(self):
        norm = self.iteration[0,:,:]**2 + self.iteration[1,:,:]**2
        inorbit = norm <= 5**2
        self.canvas[0,inorbit] += 1
        pass
    # Generate one fractal set and update self.canvas
    # def generate_set(self,power,total_iteration=1000):
    def generate_set(self,power):
        # Distance on complex plane between two pixels
        # distance_x = (self.real_p+self.real_n)/(self.px_x-1)
        # distance_y = (self.imag_p+self.imag_n)/(self.px_y-1)
        # print(distance_x,distance_y)
        # Loop through each pixel, sample random points in higher resolution
        # and average to get iteration value at that pixel
        t = TOTAL_ITERATION
        while t > 0:
            t -= 1
            # print(t)
            # for i in range(0,self.px_y):
            #     for j in range(0,self.px_x):
            #         pix = self.plane[:,i,j]
            #         prev_iter = self.iteration[:,i,j]
            #         self.iteration[:,i,j] = self.supersample_polynomial(power,pix,[distance_x,distance_y],prev_iter)
            self.polynomial(power)
            self.count_iteration()
        pass

    # SAVING IMAGES
    def save_frame(self,frame_count):
        # test = self.canvas.cpu()
        # print("max", test.max())
        # # Normalize iteration count
        # iter = self.iteration
        # print(iter)
        # print(iter[0,:,:]*iter[0,:,:])
        # # log_zn = torch.log(iter[0,:,:]*iter[0,:,:]+iter[1,:,:]*iter[1,:,:])/2.
        # nu = torch.log(log_zn/np.log(2))/np.log(2)
        # iter = iter + 1 - nu
        # test = iter
        # print(self.iteration)
        # print(self.iteration.unique())
        convert_image = torchvision.transforms.Compose([
            torchvision.transforms.ToPILImage(),
            torchvision.transforms.Grayscale(),
            torchvision.transforms.ToTensor()
        ])

        test = 255*self.canvas.cpu()/TOTAL_ITERATION
        test = convert_image(test)
        # torchvision.utils.save_image(test, "/output/test"+str(frame_count)+".jpg")
        torchvision.utils.save_image(test, "test/test"+str(frame_count)+".jpg")
        # RESET
        # Value of iteration starting at z0 = 0+0i
        self.iteration = torch.tensor((), dtype=torch.float32).new_zeros((2,self.px_y,self.px_x)).to(device)
        # Counting iterations
        self.canvas = torch.tensor((), dtype=torch.float32).new_zeros((1,self.px_y,self.px_x))
        pass


    # TRANSFORMATIONS ON COMPLEX PLANE
    # Rotate plane (need to rotate about self.center)
    # About origin:
    # z = x+yi -> exp(it)z = (xcos(t)-ysin(t)) + (xsin(t)+ycos(t))
    # About c:
    # exp(it)(z-c) + c
    def rotate_plane(self,angle=0):
        plane_real = self.plane[0,:,:].clone() - self.real_center
        plane_imag = self.plane[1,:,:].clone() - self.imag_center
        next_plane_real = plane_real*np.cos(angle)-plane_imag*np.sin(angle)
        next_plane_imag = plane_real*np.sin(angle)+plane_imag*np.cos(angle)
        self.plane[0,:,:] = next_plane_real + self.real_center
        self.plane[1,:,:] = next_plane_imag + self.imag_center
        del plane_real, plane_imag, next_plane_real, next_plane_imag
        pass

    def increment_angle(self,angle=0):
        # print("increment ", self.angle, " by ", angle)
        self.angle += angle
        self.angle = np.mod(self.angle,2*np.pi)
        # print("angle", self.angle)
        pass

    # Set plane for zooming in
    def set_plane(self,center,real,imag):
        self.plane = torch.tensor([
            [np.linspace(center[0]-real[1],center[0]+real[0],num=self.px_x)
                for i in np.arange(self.px_y)],
            [np.repeat(i,self.px_x)
                for i in np.linspace(center[1]+imag[1],center[1]-imag[0],num=self.px_y)]
        ], dtype=torch.float32).to(device)
        self.real_center = center[0]
        self.imag_center = center[1]
        self.real_p = real[0]
        self.real_n = real[1]
        self.imag_p = imag[0]
        self.imag_n = imag[1]
        # should update _n, _p as well but not used later
        self.rotate_plane(self.angle)
        pass

# FractalZoom: generate a series of fractals
class FractalZoom:
    def __init__(self, power, center, real, imag, px_x=1280, px_y=720,frame_count=0,angle=0):
        # Polynomial power
        self.power = power
        # Image framing
        self.center = center
        self.real = real
        self.imag = imag
        self.px_x = px_x
        self.px_y = px_y
        # Frame counter
        self.frame_count = frame_count
        # ComplexPlane
        self.fractal = ComplexPlane(self.center, self.real, self.imag, self.px_x, self.px_y)
        self.fractal.angle = angle
        pass

    def print_properties(self):
        print("  center: ", self.center)
        print("  real: ", self.real)
        print("  imag: ", self.imag)
        print("  angle: ", self.fractal.angle)
        print("  frame: ", self.frame_count)
        pass

    def generate_frame(self,):
        self.frame_count += 1
        if np.mod(self.frame_count,FRAMES_PER_SECOND)==0:
            print("Printing frame ", self.frame_count)
        # print(" Mem cached ", torch.cuda.memory_cached())
        # print(" Mem alloc ", torch.cuda.memory_allocated())
        self.fractal.generate_set(self.power)
        self.fractal.save_frame(self.frame_count)
        pass

    # Update image framing to prepare for next fractal set generation
    # zoom : [[real_p,real_n],[imag_p,imag_n]]
    def transform(self, new_center=None, rot_theta=None, zoom=None, seconds=0):
        frames = seconds*FRAMES_PER_SECOND
        old_center = self.center
        if new_center is None:
            new_center = self.center
        # SHOULD USE LOG SINCE EXPONENTIAL ZOOM
        # translation = np.linspace(0,1,frames)
        translation = np.linspace(0,np.pi/2,frames)
        translation = np.sin(translation)
        if rot_theta is None:
            rot_theta = 0
        angle = rot_theta/frames
        if zoom is None:
            zoom = [self.real,self.imag]
        old_real = self.real
        old_imag = self.imag
        new_real = zoom[0]
        new_imag = zoom[1]
        for f in np.arange(0,frames):
            f = np.floor(f).astype(int) #for partial seconds
            # Translation
            t = translation[f]
            z = np.add(np.multiply(old_center,1-t), np.multiply(new_center,t))
            self.center = z
            # Zoom
            self.real = np.add(old_real,np.multiply(t,np.subtract(new_real,old_real)))
            self.imag = np.add(old_imag,np.multiply(t,np.subtract(new_imag,old_imag)))
            # Rotation: NEED TO ROTATE ABOUT CENTER
            self.fractal.increment_angle(angle)
            # Construct plane
            self.fractal.set_plane(self.center,self.real,self.imag)
            # Generate fractal set and save a frame
            self.generate_frame()
        pass
        self.print_properties()

def jee():
    fz = FractalZoom(power=3,center=[0.0043108017656998815,-1.0930811521624153],real=[1,1],imag=[1,1],px_x=1280,px_y=720)    
    fz.transform(new_center=[0.0043108017656998815,1.0930811521624153],
                    rot_theta=-5*np.pi,
                    zoom=np.multiply([[1,1],[1,1]],5e-1).tolist(),
                    seconds=6.5)
    fz.transform(new_center=[0.0043108017656998815,-1.0930811521624153],
                    rot_theta=5*np.pi,
                    zoom=None,
                    seconds=6.5)
    # 13
    fz.transform(new_center=[-0.2625640379608631,-1.2619680999972],
                    rot_theta=8*np.pi,
                    zoom=np.multiply([[1,1],[1,1]],3e-2).tolist(),
                    seconds=12)
    # 25
    fz.transform(new_center=[-0.22325971185998258,-1.2186848522931981],
                    rot_theta=6*np.pi,
                    zoom=np.multiply([[1,1],[1,1]],7e-3).tolist(),
                    seconds=7)
    fz.transform(new_center=None,
                    rot_theta=8*np.pi,
                    zoom=np.multiply([[1,1],[1,1]],1e-3).tolist(),
                    seconds=6)
    # 38
    fz.transform(new_center=None,
                    rot_theta=6*np.pi,
                    zoom=np.multiply([[1,1],[1,1]],5e-3).tolist(),
                    seconds=1)
    fz.transform(new_center=None,
                    rot_theta=6*np.pi,
                    zoom=np.multiply([[1,1],[1,1]],1e-2).tolist(),
                    seconds=4)
    # 43
    fz.transform(new_center=[-0.2625640379608631,-1.2619680999972],
                rot_theta=-2*np.pi,
                zoom=np.multiply([[1,1],[1,1]],3e-2).tolist(),
                seconds=2)
    fz.transform(new_center=None,
                    rot_theta=-3*np.pi,
                    zoom=np.multiply([[1,1],[1,1]],5e-2).tolist(),
                    seconds=3)
    fz.transform(new_center=None,
                    rot_theta=2*np.pi,
                    zoom=np.multiply([[1,1],[1,1]],5e-3).tolist(),
                    seconds=3)
    # 51
    fz.transform(new_center=[-0.26375164955496755,-1.260989861825272],
                    rot_theta=-2*np.pi,
                    zoom=np.multiply([[1,1],[1,1]],7e-2).tolist(),
                    seconds=1)
    fz.transform(new_center=None,
                    rot_theta=-4*np.pi,
                    zoom=np.multiply([[1,1],[1,1]],3e-2).tolist(),
                    seconds=4)
    # 56
    fz.transform(new_center=[-0.2509738099411983,-1.2739403749473353],
                    rot_theta=2*np.pi,
                    zoom=np.multiply([[1,1],[1,1]],1e-2).tolist(),
                    seconds=2)
    fz.transform(new_center=[-0.28084632687609123,-1.2508021248359154],
                    rot_theta=-2*np.pi,
                    zoom=np.multiply([[1,1],[1,1]],1e-2).tolist(),
                    seconds=3)
    fz.transform(new_center=[-0.26375164955496755,-1.260989861825272],
                    rot_theta=-2*np.pi,
                    zoom=np.multiply([[1,1],[1,1]],1e-3).tolist(),
                    seconds=3)

    

import time
t = time.process_time()
jee()
elapsed_time = time.process_time() - t
print('Elapsed time: ',elapsed_time)

# def test(center,zoom):
#     fz = FractalZoom(3,center,zoom[0],zoom[1],1280,720)
#     fz.generate_frame()
# test([-0.2539130139508776,-1.2483473336653068],np.multiply([[1,1],[1,1]],1e-4).tolist())
