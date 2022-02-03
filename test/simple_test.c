#include "simpletest/simpletest.h"

void MyTest() { TEST(2 * 2 == 5 - 1); }

int main() {
  TEST(2 == 1 + 1);
  TEST(2 == 2);
  MyTest();
  PrintTestResult();
  return TestResult();
}
