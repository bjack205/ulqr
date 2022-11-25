add_library(lqrdata
  lqr_data.h
  lqr_data.c
  )
target_link_libraries(lqrdata
  PUBLIC
  slap
  )

add_target_to_install(lqrdata)