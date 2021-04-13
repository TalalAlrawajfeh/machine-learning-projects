@echo off

if exist .\data\train rmdir .\data\train /s /q
if exist .\data\validation rmdir .\data\validation /s /q
if exist .\data\test rmdir .\data\test /s /q

if exist *.h5 del *.h5 /f /q
