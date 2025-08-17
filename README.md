# download
hf download backseollgi/HAWK_ICCE2025 --repo-type dataset --local-dir .

# HAWK_bench 복원
cat HAWK_bench.tar.gz.part-* > HAWK_bench.tar.gz
tar -I pigz -xvf HAWK_bench.tar.gz

# HAWK_bench_json 복원
cat HAWK_bench_json.tar.gz.part-* > HAWK_bench_json.tar.gz
tar -I pigz -xvf HAWK_bench_json.tar.gz


