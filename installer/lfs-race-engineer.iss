; Inno Setup script for LFS Race Engineer
; Build with:  iscc installer\lfs-race-engineer.iss
; Output:      installer\Output\lfs-race-engineer-setup-<ver>.exe

#define MyAppName       "LFS Race Engineer"
#define MyAppShortName  "lfs-race-engineer"
#define MyAppVersion    "0.3.1"
#define MyAppPublisher  "LFS Race Engineer"
#define MyAppExeName    "lfs-race-engineer.exe"
#define MyAppSourceDir  "..\dist\lfs-race-engineer"

[Setup]
AppId={{B7A4F3D2-9E81-4C57-A2D3-7F8B1C5E6A40}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
AppVerName={#MyAppName} {#MyAppVersion}
DefaultDirName={autopf}\{#MyAppShortName}
DefaultGroupName={#MyAppName}
DisableProgramGroupPage=yes
DisableDirPage=no
AlwaysShowDirOnReadyPage=yes
UsePreviousAppDir=yes
DirExistsWarning=no
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog commandline
ArchitecturesInstallIn64BitMode=x64compatible
Compression=lzma2/max
SolidCompression=yes
WizardStyle=modern
OutputDir=Output
OutputBaseFilename={#MyAppShortName}-setup-{#MyAppVersion}
SetupIconFile=..\assets\icon.ico
UninstallDisplayName={#MyAppName} {#MyAppVersion}
UninstallDisplayIcon={app}\{#MyAppExeName}

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"
Name: "spanish"; MessagesFile: "compiler:Languages\Spanish.isl"

[Tasks]
Name: "desktopicon";    Description: "{cm:CreateDesktopIcon}";    GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked
Name: "quicklaunchicon"; Description: "{cm:CreateQuickLaunchIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked

[Files]
; The whole onedir build (.exe + every PyInstaller-emitted file +
; bundled config/, racing_lines/, tracks/ at the top level — see
; ``contents_directory='.'`` in lfs-race-engineer.spec).
Source: "{#MyAppSourceDir}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Dirs]
; Default workspace folder, writable by the user (per-user install).
Name: "{app}\captures"

[Icons]
Name: "{group}\{#MyAppName}";           Filename: "{app}\{#MyAppExeName}"; WorkingDir: "{app}"
Name: "{group}\Captures folder";        Filename: "{app}\captures"
Name: "{group}\Uninstall {#MyAppName}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\{#MyAppName}";     Filename: "{app}\{#MyAppExeName}"; WorkingDir: "{app}"; Tasks: desktopicon
Name: "{userappdata}\Microsoft\Internet Explorer\Quick Launch\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; WorkingDir: "{app}"; Tasks: quicklaunchicon

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "{cm:LaunchProgram,{#StringChange(MyAppName, '&', '&&')}}"; Flags: nowait postinstall skipifsilent unchecked

[UninstallDelete]
; Runtime log; captures/ left in place so the user keeps recorded stints.
Type: files;       Name: "{app}\studio.log"
Type: dirifempty;  Name: "{app}\captures"
Type: dirifempty;  Name: "{app}"
