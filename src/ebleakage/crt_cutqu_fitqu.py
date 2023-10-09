import numpy as np
import matplotlib.pyplot as plt
import healpy as hp


lmax = 4000
nside = 2048
cmb = np.load('./cmbdata/cmbtqunoB20482.npy')
bin_mask = np.load('./circle_mask2048.npy')
apo_mask = np.load('./apo_circle_mask2048C1_12.npy')


class EBCorrection:
    def __init__(self, m, lmax, nside, mask, post_mask, fit_flag='B'):
        ''' m are IQU maps '''
        self.m = m
        self.lmax = lmax
        self.nside = nside
        self.mask = mask
        self.post_mask = post_mask
        self.fit_flag = fit_flag

        # self.full_alm = hp.map2alm(self.m, self.lmax)
        # self.full_Tmap, self.full_Emap, self.full_Bmap = [hp.alm2map(alm, nside=self.nside) for alm in self.full_alm]
        # self.input_B = self.add_mask(self.full_Bmap)

    def add_mask(self, m):
        return m * self.mask

    def add_post_mask(self, m):
        return m * self.post_mask

    def calc_EB_fml(self, lmax, nside):
        ''' from IQU to get P_E and P_B which are the EB family '''
        masked_m = self.add_mask(self.m)
        alms = hp.map2alm(masked_m, lmax=lmax)
        alm_T, alm_E, alm_B = [alm for alm in alms]
        self.QU_E1 = self.add_mask(hp.alm2map([alm_T, alm_E, np.zeros_like(alm_T)], nside=nside))
        if self.fit_flag == 'QU':
            self.QU_B1 = self.add_mask(hp.alm2map([alm_T, np.zeros_like(alm_T), alm_B], nside=nside))
        elif self.fit_flag == 'B':
            self.corrupted_B = self.add_mask(hp.alm2map(alm_B, nside=nside))
        else:
            print('fit flag is wrong!')

    def calc_template_fml(self, lmax, nside):
        alm_QUE1 = hp.map2alm(self.QU_E1, lmax=lmax)
        alm_T, alm_E, alm_B = [alm for alm in alm_QUE1]
        if self.fit_flag == 'QU':
            self.QU_B2 = self.add_mask(hp.alm2map([alm_T,np.zeros_like(alm_T),alm_B], nside=nside))
        elif self.fit_flag == 'B':
            self.template_B = self.add_mask(hp.alm2map(alm_B, nside=nside))

    def fit_QU_B(self, lmax, nside):
        coeffs = np.polyfit(self.QU_B2[1:2].flatten(), self.QU_B1[1:2].flatten(), 1)
        slope, intercept = coeffs
        print(f"Slope: {slope}, Intercept: {intercept}")
        cleaned_QU = self.QU_B1 - slope * self.QU_B2
        self.cleaned_B = self.add_post_mask(hp.alm2map(hp.map2alm(self.add_mask(cleaned_QU), lmax=lmax)[2], nside=nside))
        self.corrupted_B = self.add_post_mask(hp.alm2map(hp.map2alm(self.add_mask(self.QU_B1), lmax=lmax)[2], nside=nside))
        self.template_B = self.add_post_mask(hp.alm2map(hp.map2alm(self.add_mask(slope * self.QU_B2), lmax=lmax)[2], nside=nside))


    def run_EBcorrection(self):
        self.calc_EB_fml(lmax=self.lmax, nside=self.nside)
        self.calc_template_fml(lmax=self.lmax, nside=self.nside)
        if self.fit_flag == 'QU':
            self.fit_QU_B(lmax=self.lmax, nside=self.nside)
        elif self.fit_flag == 'B':
            self.fit_B(lmax=self.lmax, nside=self.nside)
        else:
            print('input fit_flag is wrong!')

    def return_maps(self):
        if self.fit_flag == 'QU':
            # return self.corrupted_QU, self.cleaned_QU, self.template_QU
            return self.corrupted_B, self.cleaned_B, self.template_B
        elif self.fit_flag == 'B':
            return self.corrupted_B, self.cleaned_B, self.template_B

# obj = EBCorrection(cmb, lmax, nside, mask=bin_mask, post_mask=bin_mask, fit_flag='B')
# obj.run_EBcorrection()
# corrupted_B, cleaned_B, template_B = obj.return_maps()

obj = EBCorrection(cmb, lmax, nside, mask=bin_mask, post_mask=apo_mask, fit_flag='QU')
obj.run_EBcorrection()
# corrupted_QU, cleaned_QU, template_QU = obj.return_maps()
corrupted_B, cleaned_B, template_B = obj.return_maps()


# hp.orthview(hp.ma(cleaned_B, badval=0), half_sky=True,sub=(2,2,1), title='cleaned', min=-0.3, max=0.3, cmap='jet', norm='hist')
# hp.orthview(hp.ma(corrupted_B, badval=0), half_sky=True,sub=(2,2,2), title='corrupted', min=-0.3, max=0.3, cmap='jet',norm='hist')
# hp.orthview(hp.ma(template_B, badval=0), half_sky=True,sub=(2,2,3), title='template', min=-0.3, max=0.3, cmap='jet',norm='hist')
# # hp.orthview(hp.ma(input_B, badval=0), half_sky=True,sub=(2,2,4), title='input', min=-0.3, max=0.3, cmap='jet' ,norm='hist')
# hp.orthview(10*hp.ma(cleaned_B - input_B, badval=0), half_sky=True,sub=(2,2,4), title='residual', min=-0.3, max=0.3, cmap='jet')
# plt.subplots_adjust(wspace=0.1, hspace=0.1)
# plt.show()

hp.orthview(hp.ma(corrupted_B, badval=0), half_sky=True,sub=(1,3,1), title='corrupted', min=-0.3, max=0.3, cmap='jet', badcolor='white')
hp.orthview(hp.ma(template_B, badval=0), half_sky=True,sub=(1,3,2), title='template', min=-0.3, max=0.3, cmap='jet', badcolor='white')
hp.orthview(10*hp.ma(cleaned_B, badval=0), half_sky=True,sub=(1,3,3), title='residual', min=-0.3, max=0.3, cmap='jet', badcolor='white')
plt.show()

# hp.orthview(hp.ma(corrupted_QU, badval=0), half_sky=True,sub=(1,3,1), title='corrupted', min=-0.3, max=0.3, cmap='jet', badcolor='white')
# hp.orthview(hp.ma(template_QU, badval=0), half_sky=True,sub=(1,3,2), title='template', min=-0.3, max=0.3, cmap='jet', badcolor='white')
# hp.orthview(10*hp.ma(cleaned_QU, badval=0), half_sky=True,sub=(1,3,3), title='residual', min=-0.3, max=0.3, cmap='jet', badcolor='white')
# plt.show()

number = 2

np.save(f'./testmethod/cutqu_fitqu/{number}/corrupted_B.npy',corrupted_B)
np.save(f'./testmethod/cutqu_fitqu/{number}/cleaned_B.npy',cleaned_B)

# np.save(f'./testnew/QU/{number}/corrupted_QU.npy',corrupted_QU)
# np.save(f'./testnew/QU/{number}/cleaned_QU.npy',cleaned_QU)


