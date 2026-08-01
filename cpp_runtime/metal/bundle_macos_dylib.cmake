if(NOT DEFINED MFQ_DYLIB OR NOT EXISTS "${MFQ_DYLIB}")
    message(FATAL_ERROR "MFQ_DYLIB is not a readable dylib: ${MFQ_DYLIB}")
endif()
if(NOT DEFINED MFQ_INSTALL_NAME_TOOL OR
   NOT EXISTS "${MFQ_INSTALL_NAME_TOOL}")
    message(FATAL_ERROR
        "MFQ_INSTALL_NAME_TOOL is unavailable: ${MFQ_INSTALL_NAME_TOOL}")
endif()

execute_process(
    COMMAND /usr/bin/otool -l "${MFQ_DYLIB}"
    OUTPUT_VARIABLE _MFQ_OTOOL_OUTPUT
    RESULT_VARIABLE _MFQ_OTOOL_RESULT
)
if(NOT _MFQ_OTOOL_RESULT EQUAL 0)
    message(FATAL_ERROR "otool failed for ${MFQ_DYLIB}")
endif()

string(REGEX MATCHALL
    "path [^\n]+ \\(offset [0-9]+\\)"
    _MFQ_RPATH_LINES
    "${_MFQ_OTOOL_OUTPUT}")
set(_MFQ_HAS_LOADER_RPATH OFF)
foreach(_MFQ_RPATH_LINE IN LISTS _MFQ_RPATH_LINES)
    string(REGEX REPLACE
        "^path (.*) \\(offset [0-9]+\\)$"
        "\\1"
        _MFQ_RPATH
        "${_MFQ_RPATH_LINE}")
    if(_MFQ_RPATH STREQUAL "@loader_path")
        set(_MFQ_HAS_LOADER_RPATH ON)
    elseif(IS_ABSOLUTE "${_MFQ_RPATH}")
        execute_process(
            COMMAND "${MFQ_INSTALL_NAME_TOOL}"
                -delete_rpath "${_MFQ_RPATH}" "${MFQ_DYLIB}"
            RESULT_VARIABLE _MFQ_DELETE_RESULT
        )
        if(NOT _MFQ_DELETE_RESULT EQUAL 0)
            message(FATAL_ERROR
                "cannot remove build RPATH ${_MFQ_RPATH} from ${MFQ_DYLIB}")
        endif()
    endif()
endforeach()

if(NOT _MFQ_HAS_LOADER_RPATH)
    execute_process(
        COMMAND "${MFQ_INSTALL_NAME_TOOL}"
            -add_rpath "@loader_path" "${MFQ_DYLIB}"
        RESULT_VARIABLE _MFQ_ADD_RESULT
    )
    if(NOT _MFQ_ADD_RESULT EQUAL 0)
        message(FATAL_ERROR
            "cannot add @loader_path RPATH to ${MFQ_DYLIB}")
    endif()
endif()
