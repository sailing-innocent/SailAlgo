target("lc-test")
    set_languages("c++20")
    set_kind("binary")
    set_exceptions("cxx")

    add_includedirs("include", {public = true})
    add_files("src/**.cpp")
    add_includedirs("src")
target_end()
