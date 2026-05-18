; Inno Setup 6.3+ script for LFS Race Engineer (v0.3.7+)
; Modern Windows installer with bilingual support, auto-update checks, and best practices.
;
; Build with:
;   iscc installer\lfs-race-engineer.iss
; Output:
;   installer\Output\lfs-race-engineer-setup-<ver>.exe

#define MyAppName       "LFS Race Engineer"
#define MyAppShortName  "lfs-race-engineer"
; Version: prefer /DMyAppVersion=x.y.z passed by the build script
; (scripts\build_app.ps1 / build_app_simple.ps1 parse pyproject.toml and
; forward the version to iscc). Hardcoded fallback below is only used when
; the installer is built by hand without that flag — keep it in sync with
; pyproject.toml when you cut a release.
#ifndef MyAppVersion
  #define MyAppVersion "0.3.7"
#endif
#define MyAppPublisher  "LFS Race Engineer Contributors"
#define MyAppURL        "https://github.com/fermga/TNFR-LFS"
#define MyAppExeName    "lfs-race-engineer.exe"
#define MyAppSourceDir  "..\dist\lfs-race-engineer"

[Setup]
; === FUNDAMENTAL ===
AppId={{B7A4F3D2-9E81-4C57-A2D3-7F8B1C5E6A40}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
AppPublisherURL={#MyAppURL}
AppSupportURL={#MyAppURL}/issues
AppUpdatesURL={#MyAppURL}/releases
AppVerName={#MyAppName} {#MyAppVersion}

; === DIRECTORIES & LAYOUT ===
; Support both per-user and per-machine installs (user decides during setup).
DefaultDirName={autopf}\{#MyAppShortName}
DefaultGroupName={#MyAppName}
AllowNoIcons=yes
DirExistsWarning=auto
AlwaysShowDirOnReadyPage=yes
UsePreviousAppDir=yes
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog commandline

; === APPEARANCE ===
WizardStyle=modern
SetupIconFile=..\assets\icon.ico
UninstallDisplayIcon={app}\{#MyAppExeName}
OutputDir=Output
OutputBaseFilename={#MyAppShortName}-setup-{#MyAppVersion}
ArchitecturesInstallIn64BitMode=x64compatible
ArchitecturesAllowed=x64compatible

; === COMPRESSION & PERFORMANCE ===
Compression=lzma2/ultra
SolidCompression=yes
InternalCompressLevel=ultra
DiskSpanning=no

; === INTERNATIONALISATION ===
ShowLanguageDialog=yes

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"
Name: "spanish"; MessagesFile: "compiler:Languages\Spanish.isl"

; === SETUP TASKS ===
[Tasks]
Name: "desktopicon";        Description: "{cm:CreateDesktopIcon}";        GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked
Name: "quicklaunchicon";    Description: "{cm:CreateQuickLaunchIcon}";    GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked
Name: "assoc_lfs_csv";      Description: "Associate .csv files (telemetry replays)";  GroupDescription: "File associations"; Flags: unchecked

; === FILES ===
[Files]
; Main application directory (PyInstaller onedir layout: exe + _internal/ + config/, racing_lines/, tracks/)
Source: "{#MyAppSourceDir}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

; README (if present)
Source: "..\README.md";     DestDir: "{app}"; Flags: ignoreversion; DestName: "README.txt"

; === DIRECTORIES ===
[Dirs]
; User-writable workspace for captures & exports
Name: "{app}\captures"
Name: "{app}\exports"

; === ICONS & SHORTCUTS ===
[Icons]
; Start Menu
Name: "{group}\{#MyAppName}";                   Filename: "{app}\{#MyAppExeName}"; WorkingDir: "{app}"; Comment: "Launch {#MyAppName}"
Name: "{group}\Captures folder";                Filename: "{app}\captures"; Comment: "Your recorded stints"
Name: "{group}\README";                         Filename: "{app}\README.txt"; Comment: "Documentation"
Name: "{group}\{#MyAppName} on GitHub";         Filename: "{#MyAppURL}"; Comment: "Visit the project"
Name: "{group}\Uninstall {#MyAppName}";         Filename: "{uninstallexe}"; Comment: "Remove {#MyAppName}"

; Desktop shortcut (if task selected)
Name: "{autodesktop}\{#MyAppName}";             Filename: "{app}\{#MyAppExeName}"; WorkingDir: "{app}"; Tasks: desktopicon; Comment: "Race Engineer overlay"

; Quick Launch (if task selected, legacy)
Name: "{userappdata}\Microsoft\Internet Explorer\Quick Launch\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; WorkingDir: "{app}"; Tasks: quicklaunchicon

; === REGISTRY ===
[Registry]
; File association for .csv (telemetry replays) — if task selected
Root: HKA; Subkey: "Software\Classes\.csv\OpenWithProgids"; ValueType: string; ValueName: "{#MyAppShortName}.Replay"; ValueData: ""; Flags: uninsdeletevalue; Tasks: assoc_lfs_csv
Root: HKA; Subkey: "Software\Classes\{#MyAppShortName}.Replay"; ValueType: string; ValueData: "{#MyAppName} Telemetry Replay"; Flags: uninsdeletekey; Tasks: assoc_lfs_csv
Root: HKA; Subkey: "Software\Classes\{#MyAppShortName}.Replay\shell\open\command"; ValueType: string; ValueData: """{app}\{#MyAppExeName}"" ""%1"""; Flags: uninsdeletekey; Tasks: assoc_lfs_csv
Root: HKA; Subkey: "Software\Classes\{#MyAppShortName}.Replay\DefaultIcon"; ValueType: string; ValueData: "{app}\{#MyAppExeName},0"; Tasks: assoc_lfs_csv

; === EXECUTION ===
[Run]
; Launch after install (unchecked by default; user can opt-in)
Filename: "{app}\{#MyAppExeName}"; Description: "Launch {#MyAppName}"; Flags: nowait postinstall skipifsilent unchecked

; === UNINSTALL CLEANUP ===
[UninstallDelete]
; Remove runtime logs and empty workspace dirs; leave captures/ and exports/ for user to keep their data.
Type: files;       Name: "{app}\studio.log"
Type: files;       Name: "{app}\telemetry.log"
Type: dirifempty;  Name: "{app}"

; === CODE ===
[Code]
(* Optional: custom validation or post-install logic *)
procedure CurPageChanged(CurPageID: Integer);
begin
  (* Placeholder for custom per-page logic *)
end;

function NextButtonClick(CurPageID: Integer): Boolean;
begin
  Result := True;
  if CurPageID = wpSelectDir then
  begin
    (* Verify disk space if needed *)
  end;
end;
