# function add_ulqr_test(name)
#
# Adds a new executable test called <name>_test.
# Assumes thes source code is in a file called <name>_test.
function (add_ulqr_test name)
  set(TEST_NAME ${name}_test)
  add_executable(${TEST_NAME}
    ${TEST_NAME}.c

    test_utils.h
    test_utils.c
  )
  target_link_libraries(${TEST_NAME}
    PRIVATE 
    simpletest 
    slap 
    riccati
    m  # math library
  )
  add_test(NAME ${TEST_NAME} COMMAND ${TEST_NAME})
endfunction()

# Build the testing library
add_library(simpletest
  simpletest/simpletest.h
  simpletest/simpletest.c
)
target_include_directories(simpletest
  PRIVATE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}>
)

# Add tests
add_ulqr_test(simple)
add_ulqr_test(matrix)
add_ulqr_test(linalg)
add_ulqr_test(lqrdata)
add_ulqr_test(knotpoint)
add_ulqr_test(riccati_solver)
add_ulqr_test(double_integrator)