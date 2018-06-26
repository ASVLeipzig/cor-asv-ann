#! /bin/sed -nf

# insert table header
1i seconds	trn-loss	val-loss		nsamples	enc-maxlen	dec-maxlen

/early stopping/{z;h;b} # init hold space for next increment

/^Epoch/{n # next line gives another data point
         s/^[^-]*-//;s/s [^:]*:/	/;s/ -[^:]*:/	/ # keep only time and loss values
	 G # get columns constant over increment
	 s/\
/	/g # replace newlines by csv delimiter (tab for easy gnuplot integration)
         p # print
	 b}
	 
/: [0-9]/{s/^.*: //; # other value
          H # append to hold space
	  b}
	 