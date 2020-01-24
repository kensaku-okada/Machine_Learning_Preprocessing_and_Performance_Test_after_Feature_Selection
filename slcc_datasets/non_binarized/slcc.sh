for f in `ls *.arff`
do
    for t in 0.5 0.6 0.7 0.8 0.9 1.0
    do
	o="out/${f}-su-${t}-lcc.arff"
	echo "slcc -i " $f " -t " $t " -s su" " -o " $o
	java -Xmx8000m -jar slcc.jar -i $f -t $t -T false -o $o
    done
done
	 
