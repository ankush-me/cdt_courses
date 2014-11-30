function ftxt=colorize(txt, color, bold, highlight)
    kset = {'gray', 'red', 'green', 'yellow', 'blue', 'magenta', 'cyan', 'white', 'crimson'};
    vset = {30,31,32,33,34,35,36,37,38};
    color2num = containers.Map(kset, vset);

    num = color2num(color);
    if highlight==true
        num = num + 10;
    end
    num = num2str(num);
    if bold==true
        ftxt = sprintf('\x1b[%sm%s\x1b[0m', [num ';' '1'], txt);
    else
        ftxt = sprintf('\x1b[%sm%s\x1b[0m', num, txt);
    end
end