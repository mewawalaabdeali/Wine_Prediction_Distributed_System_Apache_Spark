#----------------------------------------------------------------
# Generated CMake target import file for configuration "Release".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "gRPC::address_sorting" for configuration "Release"
set_property(TARGET gRPC::address_sorting APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(gRPC::address_sorting PROPERTIES
  IMPORTED_IMPLIB_RELEASE "${_IMPORT_PREFIX}/lib/address_sorting.lib"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/bin/address_sorting.dll"
  )

list(APPEND _cmake_import_check_targets gRPC::address_sorting )
list(APPEND _cmake_import_check_files_for_gRPC::address_sorting "${_IMPORT_PREFIX}/lib/address_sorting.lib" "${_IMPORT_PREFIX}/bin/address_sorting.dll" )

# Import target "gRPC::gpr" for configuration "Release"
set_property(TARGET gRPC::gpr APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(gRPC::gpr PROPERTIES
  IMPORTED_IMPLIB_RELEASE "${_IMPORT_PREFIX}/lib/gpr.lib"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/bin/gpr.dll"
  )

list(APPEND _cmake_import_check_targets gRPC::gpr )
list(APPEND _cmake_import_check_files_for_gRPC::gpr "${_IMPORT_PREFIX}/lib/gpr.lib" "${_IMPORT_PREFIX}/bin/gpr.dll" )

# Import target "gRPC::grpc" for configuration "Release"
set_property(TARGET gRPC::grpc APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(gRPC::grpc PROPERTIES
  IMPORTED_IMPLIB_RELEASE "${_IMPORT_PREFIX}/lib/grpc.lib"
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "c-ares::cares;re2::re2"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/bin/grpc.dll"
  )

list(APPEND _cmake_import_check_targets gRPC::grpc )
list(APPEND _cmake_import_check_files_for_gRPC::grpc "${_IMPORT_PREFIX}/lib/grpc.lib" "${_IMPORT_PREFIX}/bin/grpc.dll" )

# Import target "gRPC::grpc_unsecure" for configuration "Release"
set_property(TARGET gRPC::grpc_unsecure APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(gRPC::grpc_unsecure PROPERTIES
  IMPORTED_IMPLIB_RELEASE "${_IMPORT_PREFIX}/lib/grpc_unsecure.lib"
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "c-ares::cares"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/bin/grpc_unsecure.dll"
  )

list(APPEND _cmake_import_check_targets gRPC::grpc_unsecure )
list(APPEND _cmake_import_check_files_for_gRPC::grpc_unsecure "${_IMPORT_PREFIX}/lib/grpc_unsecure.lib" "${_IMPORT_PREFIX}/bin/grpc_unsecure.dll" )

# Import target "gRPC::upb_base_lib" for configuration "Release"
set_property(TARGET gRPC::upb_base_lib APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(gRPC::upb_base_lib PROPERTIES
  IMPORTED_IMPLIB_RELEASE "${_IMPORT_PREFIX}/lib/upb_base_lib.lib"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/bin/upb_base_lib.dll"
  )

list(APPEND _cmake_import_check_targets gRPC::upb_base_lib )
list(APPEND _cmake_import_check_files_for_gRPC::upb_base_lib "${_IMPORT_PREFIX}/lib/upb_base_lib.lib" "${_IMPORT_PREFIX}/bin/upb_base_lib.dll" )

# Import target "gRPC::upb_json_lib" for configuration "Release"
set_property(TARGET gRPC::upb_json_lib APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(gRPC::upb_json_lib PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "C"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/upb_json_lib.lib"
  )

list(APPEND _cmake_import_check_targets gRPC::upb_json_lib )
list(APPEND _cmake_import_check_files_for_gRPC::upb_json_lib "${_IMPORT_PREFIX}/lib/upb_json_lib.lib" )

# Import target "gRPC::upb_mem_lib" for configuration "Release"
set_property(TARGET gRPC::upb_mem_lib APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(gRPC::upb_mem_lib PROPERTIES
  IMPORTED_IMPLIB_RELEASE "${_IMPORT_PREFIX}/lib/upb_mem_lib.lib"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/bin/upb_mem_lib.dll"
  )

list(APPEND _cmake_import_check_targets gRPC::upb_mem_lib )
list(APPEND _cmake_import_check_files_for_gRPC::upb_mem_lib "${_IMPORT_PREFIX}/lib/upb_mem_lib.lib" "${_IMPORT_PREFIX}/bin/upb_mem_lib.dll" )

# Import target "gRPC::upb_message_lib" for configuration "Release"
set_property(TARGET gRPC::upb_message_lib APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(gRPC::upb_message_lib PROPERTIES
  IMPORTED_IMPLIB_RELEASE "${_IMPORT_PREFIX}/lib/upb_message_lib.lib"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/bin/upb_message_lib.dll"
  )

list(APPEND _cmake_import_check_targets gRPC::upb_message_lib )
list(APPEND _cmake_import_check_files_for_gRPC::upb_message_lib "${_IMPORT_PREFIX}/lib/upb_message_lib.lib" "${_IMPORT_PREFIX}/bin/upb_message_lib.dll" )

# Import target "gRPC::upb_textformat_lib" for configuration "Release"
set_property(TARGET gRPC::upb_textformat_lib APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(gRPC::upb_textformat_lib PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "C"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/upb_textformat_lib.lib"
  )

list(APPEND _cmake_import_check_targets gRPC::upb_textformat_lib )
list(APPEND _cmake_import_check_files_for_gRPC::upb_textformat_lib "${_IMPORT_PREFIX}/lib/upb_textformat_lib.lib" )

# Import target "gRPC::utf8_range_lib" for configuration "Release"
set_property(TARGET gRPC::utf8_range_lib APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(gRPC::utf8_range_lib PROPERTIES
  IMPORTED_IMPLIB_RELEASE "${_IMPORT_PREFIX}/lib/utf8_range_lib.lib"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/bin/utf8_range_lib.dll"
  )

list(APPEND _cmake_import_check_targets gRPC::utf8_range_lib )
list(APPEND _cmake_import_check_files_for_gRPC::utf8_range_lib "${_IMPORT_PREFIX}/lib/utf8_range_lib.lib" "${_IMPORT_PREFIX}/bin/utf8_range_lib.dll" )

# Import target "gRPC::grpc++" for configuration "Release"
set_property(TARGET gRPC::grpc++ APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(gRPC::grpc++ PROPERTIES
  IMPORTED_IMPLIB_RELEASE "${_IMPORT_PREFIX}/lib/grpc++.lib"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/bin/grpc++.dll"
  )

list(APPEND _cmake_import_check_targets gRPC::grpc++ )
list(APPEND _cmake_import_check_files_for_gRPC::grpc++ "${_IMPORT_PREFIX}/lib/grpc++.lib" "${_IMPORT_PREFIX}/bin/grpc++.dll" )

# Import target "gRPC::grpc++_alts" for configuration "Release"
set_property(TARGET gRPC::grpc++_alts APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(gRPC::grpc++_alts PROPERTIES
  IMPORTED_IMPLIB_RELEASE "${_IMPORT_PREFIX}/lib/grpc++_alts.lib"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/bin/grpc++_alts.dll"
  )

list(APPEND _cmake_import_check_targets gRPC::grpc++_alts )
list(APPEND _cmake_import_check_files_for_gRPC::grpc++_alts "${_IMPORT_PREFIX}/lib/grpc++_alts.lib" "${_IMPORT_PREFIX}/bin/grpc++_alts.dll" )

# Import target "gRPC::grpc++_error_details" for configuration "Release"
set_property(TARGET gRPC::grpc++_error_details APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(gRPC::grpc++_error_details PROPERTIES
  IMPORTED_IMPLIB_RELEASE "${_IMPORT_PREFIX}/lib/grpc++_error_details.lib"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/bin/grpc++_error_details.dll"
  )

list(APPEND _cmake_import_check_targets gRPC::grpc++_error_details )
list(APPEND _cmake_import_check_files_for_gRPC::grpc++_error_details "${_IMPORT_PREFIX}/lib/grpc++_error_details.lib" "${_IMPORT_PREFIX}/bin/grpc++_error_details.dll" )

# Import target "gRPC::grpc++_reflection" for configuration "Release"
set_property(TARGET gRPC::grpc++_reflection APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(gRPC::grpc++_reflection PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "CXX"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/grpc++_reflection.lib"
  )

list(APPEND _cmake_import_check_targets gRPC::grpc++_reflection )
list(APPEND _cmake_import_check_files_for_gRPC::grpc++_reflection "${_IMPORT_PREFIX}/lib/grpc++_reflection.lib" )

# Import target "gRPC::grpc++_unsecure" for configuration "Release"
set_property(TARGET gRPC::grpc++_unsecure APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(gRPC::grpc++_unsecure PROPERTIES
  IMPORTED_IMPLIB_RELEASE "${_IMPORT_PREFIX}/lib/grpc++_unsecure.lib"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/bin/grpc++_unsecure.dll"
  )

list(APPEND _cmake_import_check_targets gRPC::grpc++_unsecure )
list(APPEND _cmake_import_check_files_for_gRPC::grpc++_unsecure "${_IMPORT_PREFIX}/lib/grpc++_unsecure.lib" "${_IMPORT_PREFIX}/bin/grpc++_unsecure.dll" )

# Import target "gRPC::grpc_authorization_provider" for configuration "Release"
set_property(TARGET gRPC::grpc_authorization_provider APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(gRPC::grpc_authorization_provider PROPERTIES
  IMPORTED_IMPLIB_RELEASE "${_IMPORT_PREFIX}/lib/grpc_authorization_provider.lib"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/bin/grpc_authorization_provider.dll"
  )

list(APPEND _cmake_import_check_targets gRPC::grpc_authorization_provider )
list(APPEND _cmake_import_check_files_for_gRPC::grpc_authorization_provider "${_IMPORT_PREFIX}/lib/grpc_authorization_provider.lib" "${_IMPORT_PREFIX}/bin/grpc_authorization_provider.dll" )

# Import target "gRPC::grpc_plugin_support" for configuration "Release"
set_property(TARGET gRPC::grpc_plugin_support APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(gRPC::grpc_plugin_support PROPERTIES
  IMPORTED_IMPLIB_RELEASE "${_IMPORT_PREFIX}/lib/grpc_plugin_support.lib"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/bin/grpc_plugin_support.dll"
  )

list(APPEND _cmake_import_check_targets gRPC::grpc_plugin_support )
list(APPEND _cmake_import_check_files_for_gRPC::grpc_plugin_support "${_IMPORT_PREFIX}/lib/grpc_plugin_support.lib" "${_IMPORT_PREFIX}/bin/grpc_plugin_support.dll" )

# Import target "gRPC::grpcpp_channelz" for configuration "Release"
set_property(TARGET gRPC::grpcpp_channelz APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(gRPC::grpcpp_channelz PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "CXX"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/grpcpp_channelz.lib"
  )

list(APPEND _cmake_import_check_targets gRPC::grpcpp_channelz )
list(APPEND _cmake_import_check_files_for_gRPC::grpcpp_channelz "${_IMPORT_PREFIX}/lib/grpcpp_channelz.lib" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
