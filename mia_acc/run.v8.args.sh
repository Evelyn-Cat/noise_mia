for idx in {0..9}
do
    python run.v8.args.py ${idx} > run${idx}.v8.sh
    # bash run${idx}.v8.sh
    wait
done

