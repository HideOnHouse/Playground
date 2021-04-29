import random

TABLE = {

}


def ip_int2str(ip: int) -> str:
    return ".".join(map(str, [ip >> 24 & 0xFF, ip >> 16 & 0xFF, ip >> 8 & 0xFF, ip & 0xFF]))


def main():
    ip = random.randint(0, 2 ** 32)


if __name__ == '__main__':
    main()
