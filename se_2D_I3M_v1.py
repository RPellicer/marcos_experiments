# This script acquires a 2D spin echo sequence 
# Basic, and not real-time.
# Assembler code does not loop but python sends the .txt for each time
# (2D phantom required, there is no slice selection on the 3rd dimension)


import scipy.fft as fft
import scipy.signal as sig

import pdb
import socket, time, warnings
import numpy as np
import matplotlib.pyplot as plt
# import scipy.fft as fft
import scipy.signal as sig
import pdb
import math

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
freq_larmor = 14.280 #♠ 14.2892 # 14.2703 # 14.2658  # 2.147 local oscillator frequency, MHz
sample_nr_echo = 64  # number of (I,Q) USEFUL samples to acquire during a shot
pe_step_nr = 4  # number of phase encoding steps

tx_dt = 0.1  # RF TX sampling dt in microseconds; is rounded to a multiple of clocks (122.88 MHz)
rx_dt = 20  # RF RX sampling dt

##### Times have to match with "<instruction_file>.txt" ####
T_tx_Rf = 100       # RF pulse length (us)
T_G_ramp_dur = 30  # Gradient ramp time
# BW = 300000          # Rf Rx Bandwidth

sample_nr_2_STOP_Seq = 256 + 5000  # Nr. of samples to acquire TO STOP the acquisition

# Correct for DC offset and scaling
scale_G_x = 0.0011 # 0.0015# 0.032
scale_G_y = 0.0011 # 0.0005
scale_G_z = 0
offset_G_x = 0 # 0.0015
offset_G_y = 0 # 0.00013
offset_G_z = 0.0

# Rf amplitude
Rf_ampl = 0.05   #0.111 # 0.07125  # for Tom

# Centering the echo
# echo_delay = 350  # us; correction for receiver delay
echo_shift_idx = 125 #97 # 95 # 92 for 50 us rx_dt

rx_dt_corr = rx_dt * 0.5  # correction factor to have correct Rx sampling time till the bug is fixed

t_G_ref_Area = rx_dt * sample_nr_echo/2
# t_G_ref_Area = ((1 / ( BW / sample_nr_echo)) / 2) * 1e6  # The time needed to do half the encoding (square pulse)
T_G_pe_dur = t_G_ref_Area + T_G_ramp_dur  # Total phase encoding gradient ON time length (us)
T_G_pre_fe_dur = t_G_ref_Area + (3/2)*T_G_ramp_dur  # Total freq. encoding REWINDER ON time length (us)
T_G_fe_dur = 2 * (t_G_ref_Area + T_G_ramp_dur)  # Total Frequency encoding gradient ON time length (us)

##### RF pulses #####
### 90 RF pulse   ###
# Time vector
t_Rf_90 = np.linspace(0, T_tx_Rf, math.ceil(T_tx_Rf / tx_dt) + 1)  # Actual TX RF pulse length
# sinc pulse
alpha = 0.46  # alpha=0.46 for Hamming window, alpha=0.5 for Hanning window
Nlobes = 1
# sinc pulse with Hamming window
# tx90 = Rf_ampl * sinc(math.pi*(t_Rf_90 - T_tx_Rf/2),T_tx_Rf,Nlobes,alpha)
tx90_clean = Rf_ampl * np.ones(np.size(t_Rf_90))

### 180 RF pulse ###
# sinc pulse
tx180 = tx90_clean * 2
tx90 = np.concatenate((tx90_clean, np.zeros(1100 - np.size(tx90_clean))))
tx180 = np.concatenate((tx180, np.zeros(2200 - np.size(tx180) - np.size(tx90))))

# For testing ONLY: shot rf to mimic echo for centering the acquisition window
tx90_echo_cent = np.hstack((tx90_clean, np.zeros(100)))

##### Gradients #####
# Phase encoding gradient shape
grad_pe_samp_nr = math.ceil(T_G_pe_dur / 10)
grad_ramp_samp_nr = math.ceil(T_G_ramp_dur / 10)
grad_pe = np.hstack([np.linspace(0, 1, grad_ramp_samp_nr),  # Ramp up
                     np.ones(grad_pe_samp_nr - 2 * grad_ramp_samp_nr),  # Top
                     np.linspace(1, 0, grad_ramp_samp_nr)])  # Ramp down
grad_pe = np.hstack([grad_pe, np.zeros(500 - np.size(grad_pe))])

# Pre-frequency encoding gradient shape
grad_pre_fe_samp_nr = math.ceil(T_G_pre_fe_dur / 10)
grad_pre_fe = np.hstack([np.linspace(0, 1, grad_ramp_samp_nr),  # Ramp up
                         np.ones(grad_pre_fe_samp_nr - 2 * grad_ramp_samp_nr),  # Top
                         np.linspace(1, 0, grad_ramp_samp_nr)])  # Ramp down
grad_pre_fe = np.hstack([grad_pre_fe, np.zeros(500 - np.size(grad_pre_fe))])

# Frequency encoding gradient shape
grad_fe_samp_nr = math.ceil(T_G_fe_dur / 10)
grad_fe = np.hstack([np.linspace(0, 1, grad_ramp_samp_nr),  # Ramp up
                     np.ones(grad_fe_samp_nr - 2 * grad_ramp_samp_nr),  # Top
                     np.linspace(1, 0, grad_ramp_samp_nr)])  # Ramp down
# sample_nr_center_G_fe = (((1 / (BW / 64)) / 2) * 1e6 + T_G_ramp_dur)/10 # Total phase encoding gradient ON time length (us)
grad_fe = np.hstack([grad_fe,
                     np.zeros(np.round(1000 - np.size(grad_fe)).astype('int'))])

# Loop repeating TR and updating the gradients waveforms
data = np.zeros([sample_nr_2_STOP_Seq, pe_step_nr], dtype=complex)
exp = Experiment(samples=sample_nr_2_STOP_Seq,  # number of (I,Q) samples to acquire during a shot of the experiment
                 lo_freq=freq_larmor,  # local oscillator frequency, MHz
                 grad_t = 3.333,
                 grad_channels=3,  # Define nr. of gradients being used
                 tx_t=tx_dt,           # RF TX sampling time in microseconds; will be rounded to a multiple of system clocks (122.88 MHz)
                 rx_t=rx_dt_corr,  # RF RX sampling time in microseconds; as above
                 instruction_file="se_2D_I3M_v1.txt")

for idx2 in range(pe_step_nr):
    exp.clear_tx()
    exp.clear_grad()
    ###### Send waveforms to RP memory ###########
    # Load the RF waveforms
    tx_idx = exp.add_tx(tx90)           # add 90 Rf data to the ocra TX memory
    tx_idx = exp.add_tx(tx180)          # add 180 Rf data to the ocra TX memory
    tx_idx = exp.add_tx(tx90_echo_cent) # add Reference pulse only for measuring acquisition delay
    scale_G_pe_sweep = 2 * (idx2 / (pe_step_nr - 1) - 0.5)  # Change phase encoding gradient magnitude

    # Adjust gradient waveforms
    grad_x_1_corr = grad_pre_fe * scale_G_x + offset_G_x
    grad_y_1_corr = grad_pe * scale_G_y * scale_G_pe_sweep + offset_G_y
    grad_z_1_corr = np.zeros(np.size(grad_x_1_corr)) + offset_G_z
    grad_x_2_corr = grad_fe * scale_G_x + offset_G_x
    grad_y_2_corr = np.zeros(np.size(grad_x_2_corr)) + offset_G_y
    grad_z_2_corr = np.zeros(np.size(grad_x_2_corr)) + offset_G_z

    # Load gradient waveforms
    grad_idx = exp.add_grad([grad_x_1_corr, grad_y_1_corr, grad_z_1_corr])
    grad_idx = exp.add_grad([grad_x_2_corr, grad_y_2_corr, grad_z_2_corr])

    # Run command to MaRCoS
    data[:, idx2] = exp.run()


# time vector for representing the received data
samples_data = len(data)
t_rx = np.linspace(0, rx_dt * samples_data, samples_data)  # us

plt.figure(1)
plt.subplot(2,1,1)
# plt.plot(t_rx, np.real(data))
# plt.plot(t_rx, np.abs(data))
# plt.plot(np.real(data))
plt.plot(np.abs(data))
plt.legend(['1', '2'])
plt.xlabel('time (us)')
plt.ylabel('signal received (V)')
plt.title('Total sampled data = %i' % samples_data)
plt.grid()

# echo_shift_idx = np.floor(echo_delay / rx_dt).astype('int')
rx_sample_nr_center_G_fe = np.round((sample_nr_echo / 2) + (T_G_ramp_dur / rx_dt)).astype('int')  # Total phase encoding gradient ON time length (us)
echo_idx = (rx_sample_nr_center_G_fe - np.floor(sample_nr_echo / 2)).astype('int') # np.floor(T_G_fe_dur / (2 * rx_dt)).astype('int') - np.floor(sample_nr_echo / 2).astype('int')
kspace = data[echo_idx + echo_shift_idx:echo_idx + echo_shift_idx + sample_nr_echo, :]

plt.subplot(2, 1, 2)
# plt.plot(t_rx[echo_idx+echo_shift_idx:echo_idx+echo_shift_idx+sample_nr_echo], np.real(kspace))
# # plt.plot(t_rx, np.imag(data))
# plt.plot(t_rx[echo_idx+echo_shift_idx:echo_idx+echo_shift_idx+sample_nr_echo], np.abs(kspace))
# plt.plot(np.real(kspace))
plt.plot(np.arange(echo_idx + echo_shift_idx,echo_idx + echo_shift_idx + sample_nr_echo, 1), np.abs(kspace))
a = np.arange(echo_idx+echo_shift_idx, echo_idx+echo_shift_idx+sample_nr_echo, 1);
plt.plot(a, np.real(kspace))
# plt.plot(a, np.abs(kspace))
plt.legend(['1', '2'])
plt.xlabel('Sample nr.')
plt.ylabel('signal received (V)')
plt.title('Echo time in acquisition from = %f' % t_rx[echo_idx + echo_shift_idx])
plt.grid()

plt.figure(2)
plt.subplot(1, 2, 1)
Y = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(kspace)))
img = np.abs(Y)
plt.imshow(np.abs(kspace), cmap='gray')
plt.title('k-Space')
plt.subplot(1, 2, 2)
plt.imshow(img, cmap='gray')
plt.title('image')
plt.show()
