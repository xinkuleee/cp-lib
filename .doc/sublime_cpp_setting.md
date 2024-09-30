```json
{
    "encoding": "utf-8",
    "working_dir": "$file_path",
    "shell_cmd": "g++ -Wall -std=c++20 \"$file_name\" -o \"$file_base_name\"",
    "file_regex": "^(..[^:]*):([0-9]+):?([0-9]+)?:? (.*)$",
    "selector": "source.c++",
   
    "variants":   
    [
        {     
        "name": "build",
            "shell_cmd": "g++ \"$file\" -o \"$file_base_name\" -g"
        },
        {
        "name": "Run",
            "shell_cmd": "g++ \"$file\" -o \"$file_base_name\" -g && start cmd /c \"\"${file_path}/${file_base_name}\" & pause\""
        },  
        {
        "name": "Run Full Stack",  
            "shell_cmd": "g++ \"$file\" -o \"$file_base_name\" -Wl,--stack=268435456 && start cmd /c \"\"${file_path}/${file_base_name}\" & pause\""  
        },  
        {
        "name": "Run with o2",  
            "shell_cmd": "g++ \"$file\" -o \"$file_base_name\" -Wl,--stack=268435456 && start cmd /c \"\"${file_path}/${file_base_name}\" & pause\""  
        },  
        {
        "name": "Open GDB",  
            "shell_cmd": "start cmd /k gdb \"${file_base_name}\""
        }
    ]
}
```

