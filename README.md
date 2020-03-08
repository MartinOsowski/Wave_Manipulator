# Wave_Manipulator
A command line based tool for time-based signal processing and playback written in Python

========READ ME=========
Wave_Manipulator is an object oriented program that allows users to generate signals objects, apply various transformations, and combine signal objects to obtain new ones. 


==Creating a New Signal==
Create a new signal object by initializing sig(freq, A). If no inputs are given for the frequency and amplitude, they are set to 1 by default. You can also specify rand=True in order to generate a random signal.

Ex. Signal_1 = sig(440)
    Signal_2 = sig(530, 2)
    Signal_3 = sig(rand = True)

==Applying Transformations==
Signal objects can be transformed by applying methods in the <sig> class. Methods in Python are called via the syntax object.method(). Below is a list of functions. 

.set_time_eq()
Brings up an input request that allows the user to set the time equation of the signal as a function of t. Mathematical functions can be used by calling the cmath or np libraries. np is not required for sin, cos, and pi.
Ex. cos(40*pi*t)/t*np.exp(t)

.set_domain(lower, upper)
Sets the domain of the signal to the [upper, lower]. If the domain is made larger, the signal will be repeated to fill the new domain.

.plot()
Plot the signal in time and frequency domains.

.phasor()
Clears the signal and asks for inputs to specify the signal as a superposition of phasors. Users specify either the phasor frequency in Hz or period in s, as well as the amplitude and phase angle. This method preserves the domain of the signal.

.stack()
Same as .phasor(), however phasors are superimposed on the existing signal. 

.stack_func()
Brings up an input request that allows the user to superimpose a new signal onto the existing signal by specifying it as a function of t. Similar to .set_time_eq() except the original signal is preserved.

.randw(fmax, amax, size, amp_seed, freq_seed, phase_seed)
Randomizes the signal by stacking several signals of random frequency, amplitude, and phase. The maximimum frequency, and maximum amplitude can be set by specifying fmax and amax respectively. The amount of signals to be stacked can be set by specifying size. Seeds for the randomization of frequency, amplitude, and phase may also be specified.

.lpass(fmax)
Applies a low pass filter to the signal with a cut-off frequency of fmax.

.hpass(fmin)
Applies a high pass filter to the signal with a cut-off frequency of fmin.

.bfilter(fmin, fmax)
Applies a band pass filter to the signal with minimum and maximum frequencies set by fmin and fmax respectively. 

.wav(filepathn)
Imports a signal from a .wav file with the specified path.

.play()
Begins playback of the signal. Requires the sounddevice library.

.stop()
Terminates playback of the signal. Requires the sounddevice library.

.rec(t)
Records from a microphone for duration t. Requires the sounddevice library.

.power()
Prints the signal power.

.modulate(cfreq)
Modulates the signal onto a carrier whose frequency is specified by cfreq.

.normalize(norm)
Normalizes the signal amplitude to norm. By default, norm=1.

.block(width, amp, start)
Creates a block signal with the specified inputs.

==Bit Signals==
<bitsig> is a subclass of the <sig> class. It is initialized as bitsig(bitlength, bitamplitude) and asks users to create binary signals by specifying a sequece of 1s and 0s. The bitlength and amplitude of a bit signal can be set when creating the object. Users can also specify a unipolar or polar signal. Bit signals have the same methods as a normal signal, and can therefor be modulated, stacked, and sequenced. 

==Combining Signals==
Signal objects can be combined to form new signals. There are three ways to do this. Signals can be multiplied with eachother, superimposed on one another, or sequenced. This is done by the following functions:

newsignal = mult(sig1, sig2)
newsignal = sup(sig1, sig2)
newsignal = seq(sig1, sig2)
