// 공식 평가 이진파일을 그대로 쓰되, 요약값 하나만 찍게 하는 구동부.
//
// evaluate_odometry.cpp 의 main 은 시험 구간 11~21 로 못박혀 있고 gnuplot·
// ps2pdf 를 system() 으로 부른다. 게다가 그 main 은 같이 배포되는 mail.h 와
// 안 맞아 그대로는 빌드도 안 된다(Mail::finalize 가 없다). 그래서 main 만
// 잘라낸 사본을 끌어와 calcSequenceErrors 를 직접 부른다. 자르는 자리는
// main 이 시작하는 줄 앞이고, 그 앞은 원본 바이트 그대로다 — 공식 코드의
// 계산 부분은 한 줄도 안 건드린다. 건드리면 대조하는 의미가 없다.
//
// 빌드와 실행은 scripts/kitti_official_binary_check.py 가 한다.
// 찍는 것: 구간 수, 회전오차 평균(라디안/미터), 병진오차 평균, 그리고 같은
// 두 평균을 saveStats 와 같은 단정밀도 누산으로 낸 값.
//
// 누산 변수만 배정밀도로 둔 것이 공식 saveStats 와의 유일한 차이라서, 그
// 차이가 얼마인지는 추정할 것이 아니라 잴 수 있다. 같은 반복문 안에서 두
// 누산을 같이 돌려 네 수를 한 번에 찍는다 — 따로 돌리면 구간 오차가 같다는
// 보장이 없다.

#include DEVKIT_FUNCS

int main(int argc, char **argv) {
  if (argc != 3) {
    fprintf(stderr, "usage: official_mean <gt_poses.txt> <result_poses.txt>\n");
    return 2;
  }
  vector<Matrix> poses_gt = loadPoses(argv[1]);
  vector<Matrix> poses_result = loadPoses(argv[2]);
  if (poses_gt.size() == 0 || poses_gt.size() != poses_result.size()) {
    fprintf(stderr, "pose count mismatch: %zu vs %zu\n",
            poses_gt.size(), poses_result.size());
    return 3;
  }
  vector<errors> err = calcSequenceErrors(poses_gt, poses_result);
  if (err.size() == 0) {
    fprintf(stderr, "no segments\n");
    return 4;
  }
  double r = 0.0, t = 0.0;
  float rf = 0.0f, tf = 0.0f;   // saveStats 의 누산 변수와 같은 형이다
  for (vector<errors>::iterator it = err.begin(); it != err.end(); it++) {
    r += it->r_err;
    t += it->t_err;
    rf += it->r_err;
    tf += it->t_err;
  }
  // 배정밀도 자리는 열일곱 자리로 찍는다. 열두 자리는 사람이 읽기 좋을 뿐
  // 되읽으면 원래 값이 아니다 — 파이썬 쪽이 이 수를 그대로 보존했다고 적는
  // 자리가 있어서, 보존이라는 말과 출력이 어긋나면 안 된다. 단정밀도 둘은
  // 아홉 자리가 왕복이라 그대로 둔다.
  printf("%zu %.17g %.17g %.9g %.9g\n", err.size(),
         r / err.size(), t / err.size(),
         rf / float(err.size()), tf / float(err.size()));
  return 0;
}
