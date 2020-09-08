conda activate fed
export PYTHONPATH=/home/xly/zf/Federated/Client
cd /home/xly/zf/Federated/Client
python Test/TestTasks/main.py &
python Test/TestTasks/data0.py &
python Test/TestTasks/data1.py &
python Test/TestTasks/label0.py &
python Test/TestTasks/computation.py &
python Test/TestTasks/crypto_producer.py &
echo "Start all test servers successfully"