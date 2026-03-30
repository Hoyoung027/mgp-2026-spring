#!/bin/bash

# 실험 설정
RUNS=10
RESULTS_FILE="benchmark_results.csv"

echo "variant,o3,run,time_sec" > $RESULTS_FILE

# 실험 함수: run_experiment <variant_name> <header_file> <o3_flag>
run_experiment() {
    local variant=$1
    local header=$2
    local o3=$3

    # 헤더 교체
    cp "$header" softmax_parallel.h

    # 빌드
    mkdir -p build
    if [ "$o3" = "yes" ]; then
        g++ -std=c++20 -O3 main.cpp -o build/main
    else
        g++ -std=c++20 main.cpp -o build/main
    fi

    echo ""
    echo "=== $variant | O3=$o3 | ${RUNS}회 실험 시작 ==="

    for i in $(seq 1 $RUNS); do
        # "parallel softmax took X sec" 라인에서 시간만 추출
        t=$(./build/main 2>/dev/null | grep "parallel softmax took" | awk '{print $4}')
        echo "$variant,$o3,$i,$t" >> $RESULTS_FILE
        printf "\r  진행: %d/%d  (last: %s sec)" $i $RUNS $t
    done
    echo ""
}

# 4가지 조합 실험
run_experiment "online"  "softmax_parallel_online.h"  "no"
run_experiment "online"  "softmax_parallel_online.h"  "yes"
run_experiment "naive"   "softmax_parallel_naive.h"   "no"
run_experiment "naive"   "softmax_parallel_naive.h"   "yes"

# 원본 복원
cp softmax_parallel_online.h softmax_parallel.h
mkdir -p build
g++ -std=c++20 main.cpp -o build/main

# 결과 요약 출력
echo ""
echo "=== 결과 요약 (단위: sec) ==="
echo "variant       | O3  | avg    | min    | max"
echo "--------------------------------------------"
for variant in online naive; do
    for o3 in no yes; do
        # awk로 avg/min/max 계산 (헤더 행 제외)
        awk -F',' -v v="$variant" -v o="$o3" '
            NR>1 && $1==v && $2==o {
                sum+=$4; count++;
                if (min=="" || $4<min) min=$4;
                if (max=="" || $4>max) max=$4;
            }
            END {
                printf "%-13s | %-3s | %.4f | %.4f | %.4f\n", v, o, sum/count, min, max
            }
        ' $RESULTS_FILE
    done
done

echo ""
echo "상세 결과: $RESULTS_FILE"