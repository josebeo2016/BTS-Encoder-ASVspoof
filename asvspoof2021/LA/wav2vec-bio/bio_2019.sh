if [ -d "./biosegment/" ] 
then
    echo "biosegment exist" 
else
    ln -s /dataa/phucdt/bio/biosegment/ .
fi
echo "start extract bio embedding"
python main.py --config $1 --track=LA --loss=CCE --lr=0.000001 --batch_size=12 --database_path=/datab/Dataset/ASVspoof/LA/ --protocols_path=/datab/Dataset/ASVspoof/LA/ --is_eval --bio_extract --model_path $2 --eval_output $3
