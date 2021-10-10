import os


def func(str):
    strs = str.split('\n')
    ans = []
    for i in range(len(strs)):
        if i % 4 == 2:
            ans.append(strs[i])
    return ans


def main():
    for root, dirs, files in os.walk('./txt2'):
        for file in files:
            print(file[5:file.find('- lang_en.srt')-1])
            filepath = os.path.join(root, file)
            with open(filepath, 'r') as f:
                str = f.read()
            ans = func(str)
            new_filepath = os.path.join('./newtxt', file)
            with open(new_filepath, 'w') as f:
                for i in range(len(ans)):
                    f.write(ans[i] + '\n')


if __name__ == '__main__':
    main()
