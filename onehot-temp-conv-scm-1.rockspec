package = "onehot-temp-conv"
version = "scm-1"

source = {
    url = "https://github.com/pengsun/onehot-temp-conv",
}

description = {
    summary = "sparse temporal convolution",
    detailed = [[
        Apply 1d convolution directly to the one-hot word vectors
    ]],
    homepage = "https://",
    license = "MIT"
}

dependencies = {
    "torch >= 7.0",
}

build = {
    type = "command",
    build_command = [[
        cmake -E make_directory build;
        cd build;
        cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="$(LUA_BINDIR)/.." -DCMAKE_INSTALL_PREFIX="$(PREFIX)";
        $(MAKE)
    ]],
    install_command = "cd build && $(MAKE) install"
}