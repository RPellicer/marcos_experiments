# This script emulates a 2D turbo spin echo sequence
# Basic, and not real-time.
# Assembler code does not loop but python sends the .txt for each time
# (2D phantom required, there is no slice selection on the 3rd dimension)


import scipy.fft as fft
import scipy.signal as sig

import pdb
import socket, time, warnings
import numpy as np
import matplotlib.pyplot as plt
# from mpldatacursor import datacursor
# import scipy.fft as fft
import scipy.signal as sig
import pdb
import math
import time

import external
from local_config import ip_address, port, fpga_clk_freq_MHz
from ocra_lib.assembler import Assembler
import server_comms as sc

from experiment import Experiment


st = pdb.set_trace

def sinc(x, tx_time, Nlobes, alpha):
    y = []
    t0 = (tx_time / 2) / Nlobes
    for ii in x:
        if ii == 0.0:
            yy = 1.0
        else:
            yy = t0 * ((1 - alpha) + alpha * np.cos(ii / Nlobes / t0)) * math.sin(ii / t0) / ii
        y = np.append(y, yy)
    return y

# Experiment parameters
freq_larmor = 2.14769  # local oscillator frequency, MHz
ETL = 2 # Echo train length
fe_resolution = 64  # number of (I,Q) USEFUL samples to acquire during a shot
pe_step_nr = 4    # number of phase encoding steps
pess_step_nr = 2    # number of phase encoding steps in the slice direction
kSpaceOrderingMode = 1 # Way kSpace is traversed durign phase encoding steps: 0 = linear, 1 = blocks center first
# Delays385
sample_nr_dig_filt = 3 # number of additional samples acquired per acquisition for filtering (digital)
# sliceSelWaitEddy = 0 # us

tx_dt = 1  # RF TX sampling dt in microseconds; is rounded to a multiple of clocks (122.88 MHz)
BW = 20000          # Rf Rx Bandwidth
rx_dt_img = (1 / BW) * 1e6 # in us, Sampling dwell time
overSamplRatio = 50 
rx_dt = rx_dt_img/overSamplRatio # in us, Sampling dwell time
dt_grad_each = 3.333
dt_grad = dt_grad_each * 3
# rx_dt = 50  # RF RX sampling dt

##### Times have to match with "<instruction_file>.txt" ####
T_tx_Rf = 100       # RF pulse length (us)
T_G_ramp_dur = 25*dt_grad  # Gradient ramp time (us)
# T_G_ramp_Rf_dur = 60  # Gradient ramp time (us)

sample_nr_2_STOP_Seq = 30000 # Nr. of samples to acquire TO STOP the acquisition

# Correct for DC offset and scaling
scale_G_ss = 0.32
scale_G_pe = 0.32
scale_G_fe = 0.32
offset_G_ss = 0.0
offset_G_pe = 0.0
offset_G_fe = 0.0

# Rf amplitude
Rf_ampl = 0.07125  # for Tom
sample_nr_echo = (fe_resolution + sample_nr_dig_filt) * overSamplRatio # number of (I,Q) TOTAL samples to acquire during a shot

# Centering the echo
# echo_delay1 = 6400  # us; correction for receiver delay
# echo_delay2 = 14150  # us; correction for receiver delay
# RxBuffIniTrash = 80 # Nr. of non-wanted samples at the begining of the RX buffer

##### RF pulses #####
### 90 RF pulse   ###
# Time vector
t_Rf_90 = np.linspace(0, T_tx_Rf, math.ceil(T_tx_Rf / tx_dt) + 1)  # Actual TX RF pulse length
# sinc pulse
# alpha = 0.46  # alpha=0.46 for Hamming window, alpha=0.5 for Hanning window
# Nlobes = 1
# sinc pulse with Hamming window
# tx90 = Rf_ampl * sinc(math.pi*(t_Rf_90 - T_tx_Rf/2),T_tx_Rf,Nlobes,alpha)
tx90_tight = Rf_ampl * np.ones(np.size(t_Rf_90)) # Hard pulse (square)
tx90 = np.concatenate((tx90_tight, np.zeros(1000 - np.size(tx90_tight))))

### 180 RF pulse ###
tx180 = tx90_tight * 2
tx180 = np.concatenate((tx180, np.zeros(2000 - np.size(tx180) - np.size(tx90))))

##### Gradients #####
t_G_ref_Area_tight = ((1 / ( BW / (fe_resolution))) / 2) * 1e6  # Time for 1/2 K space (square pulse)
t_G_ref_Area_filter = (rx_dt * sample_nr_echo) / 2  # Time for 1/2 K space with spare samples for filter (square pulse)
# t_G_sliceSel90 = T_tx_Rf + sliceSelWaitEddy * 1e6 + 2*T_G_ramp_Rf_dur # Time for 90 deg slice sel
# t_G_sliceSel180 = t_G_sliceSel90 # Time for refocusing (180 deg) slice sel
T_G_pe_dur = t_G_ref_Area_tight + T_G_ramp_dur  # Total phase encoding gradient ON time length (us)
T_G_pre_fe_dur = t_G_ref_Area_filter + (3/2)*T_G_ramp_dur  # Total freq. encoding REWINDER ON time length (us)
T_G_pre_fe_dur = math.ceil(T_G_pre_fe_dur / dt_grad)* dt_grad # to make it integer of gradient time steps so AreaPreFE = AreaFE
# T_G_fe_dur = 2 * (t_G_ref_Area_filter + T_G_ramp_dur)  # Total Frequency encoding gradient ON time length (us)
T_G_fe_dur = 2*(T_G_pre_fe_dur - T_G_ramp_dur) + T_G_ramp_dur  # Total Frequency encoding gradient ON time length (us)

# Phase encoding (dimension 1 & 2) gradient shape
grad_ramp_samp_nr = math.ceil(T_G_ramp_dur / dt_grad)
grad_pe_samp_nr = math.ceil(T_G_pe_dur / dt_grad)
grad_pe = np.hstack([np.linspace(0, 1, grad_ramp_samp_nr),  # Ramp up
                     np.ones(grad_pe_samp_nr - 2 * grad_ramp_samp_nr),  # Top
                     np.linspace(1, 0, grad_ramp_samp_nr)])  # Ramp down
grad_pe = np.hstack([grad_pe, np.zeros(190 - np.size(grad_pe))])

# Pre-frequency encoding gradient shape
grad_pre_fe_samp_nr = math.ceil(T_G_pre_fe_dur / dt_grad)
ru_Gpre_fe = np.linspace(0, 1, grad_ramp_samp_nr+1)  # Ramp up
ru_Gpre_fe = ru_Gpre_fe[0:-1] + ru_Gpre_fe[1]/2
rd_Gpre_fe = np.linspace(1, 0, grad_ramp_samp_nr+1)  # Ramp down
rd_Gpre_fe = rd_Gpre_fe[0:-1] - rd_Gpre_fe[-2]/2
grad_pre_fe = np.hstack([ru_Gpre_fe,  # Ramp up
                         np.ones(grad_pre_fe_samp_nr - 2 * grad_ramp_samp_nr),  # Top
                         rd_Gpre_fe])  # Ramp down
grad_pre_fe = np.hstack([grad_pre_fe, np.zeros(220 - np.size(grad_pre_fe))])

# Frequency encoding gradient shape
grad_fe_samp_nr = math.ceil(T_G_fe_dur / dt_grad)
ru_G_fe = np.linspace(0, 1, grad_ramp_samp_nr+1)  # Ramp up
ru_G_fe = ru_G_fe[0:-1] + ru_G_fe[1]/2
rd_G_fe = np.linspace(1, 0, grad_ramp_samp_nr+1)  # Ramp down
rd_G_fe = rd_G_fe[0:-1] - rd_G_fe[-2]/2
grad_fe = np.hstack([ru_G_fe,  # Ramp up
                     np.ones(grad_fe_samp_nr - 2 * grad_ramp_samp_nr),  # Top
                     rd_G_fe])  # Ramp down
sample_nr_center_G_fe = (((1 / (BW / 140)) / 2) * 1e6 + T_G_ramp_dur)/dt_grad # Total phase encoding gradient ON time length (us)
grad_fe = np.hstack([grad_fe, np.zeros(np.round(400 - np.size(grad_fe)).astype('int'))])

# Arrange kSpace filling
scale_G_pe_range = np.linspace(-1, 1, pe_step_nr)
scale_G_pess_range = np.linspace(-1, 1, pess_step_nr)
# to acquire non-consecutive kSpace lines
TR_nr = np.floor(pe_step_nr / ETL).astype(int)
kIdxTmp2 = np.arange(pe_step_nr)
if kSpaceOrderingMode == 0: 
    kIdxTmp2[0:np.floor(pe_step_nr / ETL).astype(int)] = np.flip(kIdxTmp2[0:np.floor(pe_step_nr / ETL).astype(int)])
    kIdxTmp2 = np.reshape(kIdxTmp2,(-1,ETL))        
elif kSpaceOrderingMode == 1: 
    kIdxTmp2 = kIdxTmp2.reshape(np.floor(TR_nr/2).astype(int),-1, order='F')
    kIdxTmp2[:,:ETL]= np.flip(kIdxTmp2[:,:ETL],0)
    kIdxTmp2[:,:ETL]= np.flip(kIdxTmp2[:,:ETL],1)
    kIdxTmp2 = np.vstack([kIdxTmp2[:,:ETL], kIdxTmp2[:,ETL:]])

# Loop repeating TR and updating the gradients waveforms
data = np.zeros([sample_nr_2_STOP_Seq, TR_nr], dtype=complex)

# Generate experiment object
exp = Experiment(samples=sample_nr_2_STOP_Seq,  # number of (I,Q) samples to acquire during a shot of the experiment
                 lo_freq=freq_larmor,  # local oscillator frequency, MHz
                 # grad_t=5.6,  # us, Gradient DAC sampling rate
                 grad_t = dt_grad_each, # 3.333,
                 grad_channels=3,  # Define nr. of gradients being used
                 tx_t=tx_dt,
                 # RF TX sampling time in microseconds; will be rounded to a multiple of system clocks (122.88 MHz)
                 rx_t=rx_dt,  # rx_dt_corr,  # RF RX sampling time in microseconds; as above
                 instruction_file="TSE_2D_tests_RX_ON.txt",  # TSE_2D_tests.txt   "TSE_2D_tests_echo_center_Rf_RX_ON.txt", #
                 assert_errors=False)

for idxTR in range(TR_nr):
    ## Initialise data buffers
    exp.clear_tx()
    exp.clear_grad()
    ###### Send waveforms to RP memory ###########
    tx_length = np.zeros(1).astype(int)
    # Load the RF waveforms
    # tx_idx = exp.add_tx(tx90.astype(complex))               # add 90x+ Rf data to the ocra TX memory
    tx_idx = exp.add_tx(tx90.astype(complex))               # add 90x+ Rf data to the ocra TX memory
    tx_length = np.hstack([tx_length, tx_length[-1] + tx90.size])
    tx_idx = exp.add_tx(tx180.astype(complex))              # add 180x+ Rf data to the ocra TX memory
    tx_length = np.hstack([tx_length, tx_length[-1] + tx180.size])
    tx_idx = exp.add_tx(tx180.astype(complex)*1j)           # add 180y+ Rf data to the ocra TX memory
    tx_length = np.hstack([tx_length, tx_length[-1] + tx180.size])
    tx_idx = exp.add_tx(tx180.astype(complex)*(-1j))        # add 180y- Rf data to the ocra TX memory
    tx_length = np.hstack([tx_length, tx_length[-1] + tx180.size])

    # Adjust gradient waveforms
    # Echo nr                  |               1               |               2               |
    # Block    |    1   |   2  |  3  |   4   |   5     |   6   |  7  |   8   |    9    |  10   |
    # Mem      |    0   |   2  |  3  |   4   |   5     |   6   |  7  |   8   |    9    |  10   |
    # RF       |_$$_____|______|_$$__|_______|_________|_______|_$$__|_______|_________|_______|_$$_
    #          |        |      |     |       |         |       |     |       |         |       |
    # Gss      |/--\   _|______|/--\_|_______|_________|_______|/--\_|_______|_________|_______|/--\
    #          |    \_/ |      |     |       |         |       |     |       |         |       |
    # Gpe      |________|______|_____|/1111\_|_________|      _|_____|/2222\_|_________|      _|____
    #          |        |      |     |       |         |\1111/ |     |       |         |\2222/ |
    # Gfe      |________|/---\_|_____|_______|/------\_|_______|_____|_______|/------\_|_______|____


    # Shape Gradients block by block
    G_length = np.zeros(1).astype(int)
    G_length_Non_Zero = np.zeros(1).astype(int)
    # Block 1: Rf90 + ss
    # Block 2: -fe/2 prephase
    grad_fe_2_corr = grad_pre_fe * scale_G_fe + offset_G_fe
    grad_pe_2_corr = np.zeros(np.size(grad_fe_2_corr)) + offset_G_pe
    grad_ss_2_corr = np.zeros(np.size(grad_fe_2_corr)) + offset_G_ss
    grad_idx = exp.add_grad([grad_ss_2_corr, grad_pe_2_corr, grad_fe_2_corr])
    G_length = np.hstack([G_length, G_length[-1] + grad_fe_2_corr.size])
    G_length_Non_Zero = np.hstack([G_length_Non_Zero, grad_pre_fe_samp_nr])

    for idxETL in range(ETL):
        scale_G_pe_sweep = scale_G_pe_range[kIdxTmp2[idxTR, idxETL]]
        # ----echo 1--------------------
        # Block 3: Rf180 + ss
        # Block 4: pe+
        grad_pe_4_corr = grad_pe * scale_G_pe * scale_G_pe_sweep + offset_G_pe
        grad_fe_4_corr = np.zeros(np.size(grad_pe_4_corr)) + offset_G_fe
        grad_ss_4_corr = np.zeros(np.size(grad_pe_4_corr)) + offset_G_ss
        grad_idx = exp.add_grad([grad_ss_4_corr, grad_pe_4_corr, grad_fe_4_corr])
        G_length = np.hstack([G_length, G_length[-1] + grad_fe_4_corr.size])
        G_length_Non_Zero = np.hstack([G_length_Non_Zero, grad_pe_samp_nr])

        # Block 5: fe
        grad_fe_5_corr = grad_fe * scale_G_fe + offset_G_fe
        grad_pe_5_corr = np.zeros(np.size(grad_fe_5_corr)) + offset_G_pe
        grad_ss_5_corr = np.zeros(np.size(grad_fe_5_corr)) + offset_G_ss
        grad_idx = exp.add_grad([grad_ss_5_corr, grad_pe_5_corr, grad_fe_5_corr])
        G_length = np.hstack([G_length, G_length[-1] + grad_fe_5_corr.size])
        G_length_Non_Zero = np.hstack([G_length_Non_Zero, grad_fe_samp_nr])
        
        # Block 6: pe-
        grad_pe_6_corr = grad_pe * (-scale_G_pe) * scale_G_pe_sweep + offset_G_pe
        grad_fe_6_corr = np.zeros(np.size(grad_pe_6_corr)) + offset_G_fe
        grad_ss_6_corr = np.zeros(np.size(grad_pe_6_corr)) + offset_G_ss
        grad_idx = exp.add_grad([grad_ss_6_corr, grad_pe_6_corr, grad_fe_6_corr])
        G_length = np.hstack([G_length, G_length[-1] + grad_fe_6_corr.size])
        G_length_Non_Zero = np.hstack([G_length_Non_Zero, grad_pe_samp_nr])

    # Run command to MaRCoS
    data[:, idxTR] = exp.run()
    time.sleep(0.1) # For testing purposes - Slow down time between TR

# time vector for representing the received data
samples_data = len(data)
t_rx = np.linspace(0, rx_dt * samples_data, samples_data)  # us

echo_shift_idx_1 = 6491  # RxBuffIniTrash + np.floor(echo_delay1 / rx_dt).astype('int')
echo_shift_idx_2 = 14519 # RxBuffIniTrash + np.floor(echo_delay2 / rx_dt).astype('int')

kspaceOver = np.zeros([sample_nr_echo, TR_nr * ETL]).astype(complex)
kspaceTmp = np.zeros([sample_nr_echo, TR_nr * ETL]).astype(complex)

kspaceTmp[:, 0::2] = data[echo_shift_idx_1:echo_shift_idx_1 + sample_nr_echo, :]
kspaceTmp[:, 1::2] = data[echo_shift_idx_2:echo_shift_idx_2 + sample_nr_echo, :]
kspaceOver[:,np.squeeze(kIdxTmp2.reshape(-1, 1))] = kspaceTmp
# kspaceOver = np.squeeze(kspaceTmp[:,kIdxTmp2.reshape(-1, 1)])
kspace = sig.decimate(kspaceOver, overSamplRatio, axis=0)

### Correct for shift on echoes  ###
tresholdWindowIdx = np.array([0, 300])
trigTres = np.argmax(abs(data[tresholdWindowIdx[0]:tresholdWindowIdx[1],:]) > 0.1, axis = 0)
trigTres = (trigTres - np.ceil(np.mean(trigTres))).astype(int)
trigTres = trigTres[..., None]
kspaceOverJitt = np.zeros([sample_nr_echo, TR_nr * ETL]).astype(complex)
kspaceTmpJitt = np.zeros([sample_nr_echo, TR_nr * ETL]).astype(complex)
tmp1 = np.arange(echo_shift_idx_1, echo_shift_idx_1 + sample_nr_echo)
tmp2 = np.arange(echo_shift_idx_2, echo_shift_idx_2 + sample_nr_echo)
echoWind1 = tmp1 + trigTres
echoWind1 = echoWind1.T
echoWind2 = tmp2 + trigTres
echoWind2 = echoWind2.T

for idxTR in range(TR_nr):
    kspaceTmpJitt[:, idxTR * ETL] = data[echoWind1[:,idxTR], idxTR]
    kspaceTmpJitt[:, idxTR * ETL + 1] = data[echoWind2[:,idxTR], idxTR]
kspaceOverJitt[:,np.squeeze(kIdxTmp2.reshape(-1, 1))] = kspaceTmpJitt
# kspaceOverJitt = np.squeeze(kspaceTmpJitt[:,kIdxTmp2.reshape(-1, 1)])
kspaceJitt = sig.decimate(kspaceOverJitt, overSamplRatio, axis=0)
timestr = time.strftime("%Y%m%d-%H%M%S")
filemane = timestr + str("outfile")
np.savez(filemane, data=data, kspace=kspace, kspaceOverJitt=kspaceOverJitt)
# np.save("testData.npy", data)

plt.figure(1)
plt.subplot(3,1,1)
# plt.plot(t_rx, np.real(data))
# plt.plot(t_rx, np.abs(data))
# plt.plot(np.real(data))
plt.plot(np.abs(data))
# datacursor(display='multiple', draggable=True)
plt.legend(['1st acq', '2nd acq'])
plt.xlabel('time (us)')
plt.ylabel('signal received (V)')
plt.title('Total sampled data = %i' % samples_data)
plt.grid()

plt.subplot(3, 1, 2)
plt.plot(np.abs(kspace))
plt.legend(['1st acq', '2nd acq','3','4','5','6','7','8'])
plt.subplot(3, 1, 3)
plt.plot(np.arange(echo_shift_idx_1, echo_shift_idx_1 + sample_nr_echo,1), np.abs(kspaceTmp[:, 0::2]))
plt.plot(np.arange(echo_shift_idx_2, echo_shift_idx_2 + sample_nr_echo,1), np.abs(kspaceTmp[:, 1::2]))
# datacursor(display='multiple', draggable=True)
plt.legend(['1st acq', '2nd acq','3','4','5','6','7','8'])
plt.xlabel('Sample nr.')
plt.ylabel('signal received (V)')
plt.title('Echo time in acquisition from = %f' % t_rx[echo_shift_idx_1])
plt.grid()

plt.figure(2)
plt.subplot(1, 3, 1)
Y = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(kspace)))
img = np.abs(Y)
plt.imshow(np.abs(kspace), cmap='gray')
plt.title('k-Space')
plt.subplot(1, 3, 2)
Y = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(kspaceJitt)))
img = np.abs(Y)
plt.imshow(np.abs(kspace), cmap='gray')
plt.title('k-Space Jitter corrected')
plt.subplot(1, 3, 3)
plt.imshow(img, cmap='gray')
plt.title('image')
plt.show()

print('&&&&&  MEMORY   &&&&& Gradient offests should start at:')
print((G_length*3))
print('Each gradient is full until')
print((G_length[:-1]*3+G_length_Non_Zero[1:]*3))

print('&&&&&  TIME   &&&&& Gradient memory can last until:')
print(((G_length[1:]-G_length[0:-1])*10))
print('Each gradient is ON for')
print((G_length_Non_Zero[1:]*10))
