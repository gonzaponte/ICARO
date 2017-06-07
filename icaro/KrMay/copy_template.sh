for i in $@
do
    name=Run${i}.ipynb
    echo Running ${name}
    cp Kr_template.ipynb ${name}
    perl -pi -e 's/XXXX/'"$i"'/g' ${name}
    jupyter nbconvert --ExecutePreprocessor.timeout=600 --to notebook --execute ${name} --output ${name}
done