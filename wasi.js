// Mock implementation of WASI snapshot_preview1 functions for the browser
export function fd_write(fd, iovs, iovs_len, nwritten) {
    return 0;
}

export function fd_read(fd, iovs, iovs_len, nread) {
    return 0;
}

export function fd_seek(fd, offset, whence, newoffset) {
    return 0;
}

export function fd_close(fd) {
    return 0;
}
