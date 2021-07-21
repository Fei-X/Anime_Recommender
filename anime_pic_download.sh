#!/bin/bash
while IFS="," read -r id url
    do
        wget="wget -O ${id}.jpg $url"
        `$wget`
    done < <(tail -n +2 try.csv)
