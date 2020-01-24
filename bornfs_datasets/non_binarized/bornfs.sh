for f in `ls *.arff`
do
    for t in 0.5 0.6 0.7 0.8 0.9 1.0
    do
	for s in ratio noise relevance
	do
	    o="out/${f}-${s}-${t}-bornfs.arff"
	    echo "bornfs2 -i " $f " -t " $t " -s " $s " -o " $o
	    java -Xmx8000m -jar bornfs2.jar -i $f -s $s -t $t -T false -o $o
	done
    done
done
	 
