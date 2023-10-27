if [ -d "./biosegment/" ] 
then
    echo "biosegment exist" 
else
    ln -s /dataa/phucdt/bio/biosegment/ .
fi
echo "start training"
python main.py --config $1 --track=LA --loss=CCE --lr=0.0001 --batch_size=64 --database_path=/dataa/Dataset/ASVspoof/LA/ --protocols_path=/dataa/Dataset/ASVspoof/LA/
