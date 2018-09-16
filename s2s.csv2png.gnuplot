#! /usr/bin/gnuplot -c

# (or just `gnuplot -c thisfile -` if you want to stay interactive)

if (ARGC < 1) {
   arg = ARG0 # from shebang
} else {
   arg = ARG1 # from gnuplot
}
datafile = arg[1:strlen(arg)-3]."csv" # remove .gnuplot from argument, add .csv (from s2s.log2csv.sed from keras log)
datafile2 = arg[1:strlen(arg)-3]."len" # remove .gnuplot from argument, add .len (byte lengths of traindata)
#datafile = "s2s.3lstm+blstm320.beam.large.h5.csv"
datapoints = system(sprintf("wc -l < %s", datafile))-1
datapoints2 = system(sprintf("wc -l < %s", datafile2))-1

#set term qt enhanced size 768,1280

set term png size 768, 1280
set output arg[1:strlen(arg)-3]."png"
#set output "s2s.3lstm+blstm320.beam.large.h5.png"


#set lmargin at screen 0.10
#set rmargin at screen 0.95
#set tmargin 0.95
#set bmargin 0.05
#set size 1.0,1.0
#set origin 0,0

set grid
set key center top
#set multiplot title 'training byte-level seq2seq model with BLSTM + 3 LSTM and 320 nodes HL for eng-fra' layout 5,1 upwards margins 0.1, 0.95, 0.1, 0.9 spacing 0.05
set multiplot title "training byte-level seq2seq model with BLSTM + 3 LSTM and 320 nodes HL\nfor ".arg[1:strlen(arg)-4] layout 5,1 upwards margins 0.1, 0.95, 0.1, 0.9 spacing 0.05


set xlabel "epoch (early stopping per batch)"
set xrange [1:datapoints] writeback # store (does not work)

# plot number of samples per batch
set ytics 5000
#set yrange [0:15000]
plot datafile using 4 with histeps title columnhead(4) # nsamples

set xtics format '' # do not repeat epoch number labels on following plots
unset xlabel


# plot training time per epoch
set ytics 100
#set yrange [0:400]
plot datafile using 1 with points pointtype 2 pointsize 0.5 title columnhead(1) # seconds


# plot training and validation loss rates
set tmargin at screen 0.75
set bmargin at screen 0.45
set ytics autofreq
set yrange [*:*]
plot for [i=2:3] datafile using i with lines dashtype i-1 title columnhead(i)

set multiplot next # leave space for overscaled previous plot


# plot sequence lengths
set ytics autofreq logscale
set logscale y 2
set yrange [10:300]
set key at graph 0.4, graph 0.95
plot for [i=6:5:-1] datafile using i with histeps dashtype i-2 title columnhead(i) # enc/dec-maxlen
set multiplot previous
set key at graph 0.6, graph 0.95
set yrange [10:300] # override so it exactly fits the previous plot
unset xtics
set xrange [1:datapoints2] # take all individual sequences (not batch epochs)
#plot "fra-eng/len.csv" using 2 smooth bezier dashtype 2 title "dec-avglen",\
#     "fra-eng/len.csv" using 1 smooth bezier dashtype 1 title "enc-avglen"
plot datafile2 using 2 smooth bezier dashtype 2 title "dec-avglen",\
     datafile2 using 1 smooth bezier dashtype 1 title "enc-avglen"
#plot "fra-eng/len.csv" every ::::140585 using 2 smooth bezier dashtype 2 title "dec-avglen",\
#     "fra-eng/len.csv" every ::::140585 using 1 smooth bezier dashtype 1 title "enc-avglen"

# things that (still!) would look awkward in next plot...
set key center top
#set xrange restore # does not work
set xrange [*:*]
set xtics scale 0.1, 0.1
set ytics nologscale
unset logscale y


unset multiplot


