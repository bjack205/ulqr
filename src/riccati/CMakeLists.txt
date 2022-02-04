add_library(riccati
  riccati_solver.h
  riccati_solver.c

  lqr_data.h
  lqr_data.c

  # riccati_solve.h
  # riccati_solve.c
)
target_link_libraries(riccati
  PUBLIC
  slap
)

add_target_to_install(riccati)