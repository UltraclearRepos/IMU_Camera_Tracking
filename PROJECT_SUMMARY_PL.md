# Śledzenie sondy USG — podsumowanie projektu i dotychczasowych wniosków

Stan na: 4 sierpnia 2026

## 1. Cel projektu

Celem jest wyznaczanie pozycji i orientacji sondy USG. Rozważane źródła danych to:

- IMU zamontowane na sondzie,
- kamera skierowana na skórę,
- markery ArUco używane do inicjalizacji lub budowy mapy,
- Dobot jako ground truth podczas eksperymentów.

Pozycja początkowa każdego nagrania jest traktowana jako `[0, 0, 0]`, a dalszy ruch jest względny do pierwszej próbki. Docelowo wynik nie powinien opisywać wyłącznie środka kamery lub IMU, lecz wybrany punkt sondy, na przykład środek jej powierzchni roboczej.

## 2. Najważniejszy ogólny wniosek

Samo IMU dobrze nadaje się do krótkotrwałego przewidywania ruchu i wyznaczania orientacji, ale bardzo źle nadaje się do długotrwałego wyznaczania pozycji bez zewnętrznej korekcji. Nawet mały błąd akcelerometru jest dwukrotnie całkowany i szybko powoduje duży dryft pozycji.

Najbardziej sensowny układ docelowy to:

- kamera lub ArUco dostarcza pozycję bez dryftu,
- IMU dostarcza szybką orientację i ruch pomiędzy pomiarami kamery,
- filtr łączy oba pomiary,
- wynik jest przeliczany ze środka kamery/IMU na właściwy punkt sondy za pomocą znanej kalibracji przestrzennej.

## 3. IMU i ESKF

### 3.1. Co zostało omówione

Wyjaśniliśmy podstawowe pojęcia:

- roll — obrót wokół osi X,
- pitch — obrót wokół osi Y,
- yaw — obrót wokół osi Z,
- bias — stałe lub wolnozmienne przesunięcie pomiaru,
- noise — szybkie losowe wahania pomiaru,
- macierz kowariancji `P` — aktualna niepewność stanu filtra oraz zależności między jego błędami.

W ESKF nominalny stan zawiera:

- pozycję,
- prędkość,
- orientację jako quaternion,
- bias akcelerometru,
- bias żyroskopu.

Macierz błędu ma 15 składowych:

```text
3 pozycja + 3 prędkość + 3 orientacja + 3 bias akcelerometru + 3 bias żyroskopu
```

Filtr najpierw wykonuje predykcję z IMU, propaguje `P`, a następnie może skorygować stan pomiarem zewnętrznym, np. ZUPT albo pozą kamery. Kalman gain wynika z porównania niepewności przewidywanego stanu i pomiaru.

### 3.2. Kalibracja IMU

W [IMU/calibrate_imu.py](IMU/calibrate_imu.py) używane są nagrania sześciu statycznych orientacji:

```text
+X, -X, +Y, -Y, +Z, -Z
```

Grawitacja dostarcza znanego wektora odniesienia dla każdej pozycji. Aktualny skrypt:

- korzysta z akcelerometru i żyroskopu drugiego IMU (`imu2_*`),
- przelicza `mg` na `m/s²`, a `mdps` na `rad/s`,
- wyznacza macierz i bias akcelerometru,
- wyznacza osobny bias każdej osi żyroskopu,
- oblicza szum każdej osi,
- zapisuje wynik w `IMU/calibration/imu_calibration.json`.

Sześć pozycji jest potrzebnych, ponieważ pozwala rozdzielić bias, różną skalę osi oraz ewentualne mieszanie osi. Pojedynczy pomiar w spoczynku wystarcza do oszacowania szumu i żyroskopowego zera w tej chwili, ale nie wystarcza do pełnej kalibracji akcelerometru.

### 3.3. Aktualny ESKF

[IMU/eskf.py](IMU/eskf.py) zawiera prosty ESKF, który:

- wczytuje wcześniejszą kalibrację z JSON,
- ma dodatkową inicjalizację z początku aktualnego nagrania,
- propaguje pozycję, prędkość i quaternion z IMU,
- estymuje biasy w stanie,
- opcjonalnie wykonuje ZUPT,
- opcjonalnie przyjmuje pozycję i orientację kamery jako korekcję,
- ma mnożniki niepewności akcelerometru i żyroskopu,
- zapisuje osobne wykresy pozycji i orientacji oraz błędy względem GT.

ZUPT oznacza aktualizację zakładającą zerową prędkość, gdy filtr pewnie wykryje bezruch. W nagraniach z ciągłym ruchem nie można na nim polegać.

### 3.4. Wynik eksperymentów z IMU

Pozycja z samego IMU nadal działała bardzo słabo. Dodanie kamery do ESKF również potrafiło pogorszyć wynik względem samej kamery. Najbardziej prawdopodobne przyczyny to:

- niedokładne biasy i szum IMU,
- przeciekanie grawitacji do przyspieszenia liniowego przez błąd orientacji,
- niewłaściwie dobrane niepewności IMU i kamery,
- brak dokładnej transformacji kamera–IMU–sonda,
- różne układy współrzędnych,
- błędy czasowe,
- korekcja filtrem pomiaru kamery, który sam ma chwilowe błędy.

ESKF nie usuwa dryftu z samego IMU „algorytmicznie”. Potrzebuje okresowego pomiaru absolutnego, ograniczeń ruchu albo rzeczywistych okresów bezruchu.

## 4. Kalibracja przestrzenna kamery, IMU i sondy

Żeby połączyć kamerę i IMU, trzeba znać ich stałą relację:

- `R_IC` — obrót układu kamery względem IMU,
- `r_IC` — przesunięcie środka kamery względem IMU,
- `R_IP` — obrót docelowego układu sondy względem IMU,
- `r_IP` — przesunięcie docelowego punktu sondy względem IMU.

Nie wystarczy zamienić znaków osi. Obrót sondy powoduje ruch punktu odsuniętego od środka obrotu, dlatego nawet mały offset może wpływać na pozycję podczas zmian orientacji.

Macierze w [Camera/recording_axes.py](Camera/recording_axes.py) służą obecnie do dopasowania konwencji osi wyników kamery do GT dla konkretnych nagrań. Są użyteczne do ewaluacji, ale nie zastępują fizycznej kalibracji extrinsic kamera–Dobot/sonda. Wyniki `LineArc-1-2cm` pokazały dwa spójne ustawienia zależne od grupy nagrań. Dla `white` zastosowano:

```text
X = -raw_camera_X
Y =  raw_camera_Z
Z =  raw_camera_Y
```

Dla `dark` zastosowano:

```text
X =  raw_camera_X
Y =  raw_camera_Z
Z = -raw_camera_Y
```

## 5. Synchronizacja danych

[synchronize.py](synchronize.py) dodaje do plików CSV kolumnę `sync_timestamp`. Zachowywany jest również surowy `timestamp`. Kamera i GT są porównywane na podstawie zsynchronizowanego czasu, bez przesuwania wynikowej trajektorii tak, aby sztucznie pasowała do GT.

Ground truth jest normalizowany przestrzennie:

- od każdej pozycji odejmowana jest pierwsza pozycja,
- orientacja jest liczona jako względny obrót `R0ᵀ · R(t)`, a nie przez proste odejmowanie kątów Eulera.

Drugie podejście jest potrzebne, ponieważ kąty Eulera zawijają się i ich składowe zależą od kolejności obrotów.

Format danych został ujednolicony dla kolejnych folderów. Dane zawierają osobne katalogi m.in. `videos`, `video_timestamps`, `imu` i `dobot`, a pliki Dobota mają także kolumny `roll`, `pitch`, `yaw`.

## 6. Rozwój trackera kamery

### 6.1. Sprawdzone podejścia

W projekcie powstały trzy główne wersje:

1. [Camera/camera_tracking_optical_flow.py](Camera/camera_tracking_optical_flow.py) — Lucas–Kanade optical flow.
2. [Camera/camera_tracking.py](Camera/camera_tracking.py) — globalna mapa, DISK i LightGlue.
3. [Camera/camera_tracking_hybrid.py](Camera/camera_tracking_hybrid.py) — LightGlue okresowo odnawia dopasowania, a pomiędzy tymi klatkami punkty śledzi optical flow.

Optical flow jest szybki, ale sam śledzi lokalne przemieszczenie pikseli i może dryfować. LightGlue jest wolniejszy, ale może ponownie dopasować bieżący obraz do zapisanej mapy. Hybryda miała połączyć szybkość optical flow z możliwością ponownej lokalizacji przez LightGlue.

Pierwotnie rozważany był SuperPoint, ale obecna główna implementacja używa DISK jako detektora i deskryptora oraz LightGlue jako matchera.

### 6.2. Aktualny pipeline DISK + LightGlue

Główna implementacja znajduje się w [Camera/skin_map_tracker.py](Camera/skin_map_tracker.py).

Kolejność działania jest następująca:

1. Ograniczenie obrazu do konfigurowalnego ROI.
2. Wykrycie punktów i deskryptorów przez DISK.
3. Inicjalizacja pozy kamery markerem ArUco.
4. Utworzenie pierwszej mapy z punktów stabilnych przez kilka klatek.
5. Wybranie globalnych landmarków, które według poprzedniej pozy powinny być obecnie widoczne.
6. Dopasowanie deskryptorów globalnej mapy do punktów bieżącej klatki przez LightGlue.
7. Utworzenie par:

```text
globalna pozycja landmarku 3D ↔ piksel w aktualnej klatce
```

8. `solvePnPRansac` wyznacza pozę kamery.
9. Inliery PnP to dopasowania zgodne geometrycznie z jedną wyznaczoną pozą.
10. `solvePnPRefineLM` poprawia pozę na inlierach.
11. Aktualizowana jest jakość użytych i odrzuconych landmarków.
12. Jeśli pokrycie mapy w obrazie jest zbyt słabe, klatka może rozszerzyć mapę.

`PnP correspondences` oznacza liczbę par 3D–2D przekazanych do PnP. `PnP inliers` oznacza podzbiór tych par zaakceptowany przez RANSAC jako zgodny z wyliczoną pozą.

### 6.3. Globalna mapa i keyframe'y

Przyjęta architektura jest podobna koncepcyjnie do systemów SLAM:

- istnieje jedna globalna mapa unikalnych landmarków,
- każdy landmark ma pozycję, deskryptor i statystyki jakości,
- keyframe przechowuje obserwacje i identyfikatory globalnych landmarków,
- ten sam landmark może występować w wielu keyframe'ach,
- nowe dopasowanie blisko istniejącego landmarku może zostać zapisane jako kolejna obserwacja zamiast duplikatu.

Keyframe nie jest niezależną mapą. Jest zbiorem obserwacji punktów należących do mapy globalnej.

### 6.4. Aktualne zasady inicjalizacji i rozszerzania mapy

Aktualne ważniejsze ustawienia w `skin_map_tracker.py`:

- do 512 cech DISK w bieżącej klatce,
- inicjalizacja przez 5 klatek,
- minimum 3 obserwacje punktu podczas inicjalizacji,
- maksimum 200 landmarków w pierwszej mapie,
- maksimum 100 nowych landmarków z jednego kolejnego keyframe'u,
- maksimum 1024 globalne landmarki,
- konfigurowalna siatka pokrycia obrazu, obecnie `6 × 6`,
- obecnie cel 7 widocznych landmarków w każdej komórce,
- rozszerzenie przy średnim pokryciu poniżej ustawionego progu, zwykle `0.70`,
- surowy próg jakości DISK `50` w niedostatecznie pokrytych polach,
- surowy próg `75` w polach już pokrytych.

Pierwsza mapa wybiera najlepsze stabilne punkty. Później priorytet otrzymują dobre punkty w komórkach, w których brakuje pokrycia. Jeżeli zostaje miejsce, bardzo dobre punkty mogą zostać dodane również w obszarach już pokrytych. Limit jest górną granicą — mapa nie musi być za każdym razem dopychana maksymalną liczbą nowych punktów.

Progi konieczne do PnP zależą zarówno od absolutnego minimum, jak i od liczby globalnych punktów, które potencjalnie można dopasować. Ma to zapobiegać stosowaniu jednej stałej liczby dla bardzo małej i bardzo dużej mapy.

### 6.5. Pruning mapy

Pruning jest wykonywany podczas dodawania nowych punktów, gdy mapa potrzebuje miejsca. Oceniane są m.in.:

- liczba sytuacji, w których punkt powinien być widoczny,
- liczba rzeczywistych dopasowań,
- liczba przypadków, gdy punkt był inlierem PnP,
- liczba keyframe'ów obserwujących punkt,
- lokalne zagęszczenie landmarków.

Nowe punkty są chwilowo chronione, żeby dostały szansę zebrać obserwacje. Przy podobnej jakości łatwiej usuwane są punkty z mocno zagęszczonych obszarów niż pojedyncze punkty zapewniające unikalne pokrycie.

### 6.6. Diagnostyka i batch

Tracker zapisuje:

- CSV pozy kamery,
- osobne wykresy pozycji i orientacji wraz z błędami składowych,
- RMSE i MAE,
- procent poprawnie śledzonych klatek,
- wykres diagnostyczny dopasowań, inlierów, pokrycia i zmian liczby landmarków,
- film diagnostyczny z kolorami punktów,
- film mapy z widokiem z góry,
- czas wykonania poszczególnych etapów.

[Camera/run_camera_tracking_batch.py](Camera/run_camera_tracking_batch.py) uruchamia serię nagrań, pozwala podawać parametry osobno dla każdego nagrania i tworzy zbiorcze zestawienie RMSE oraz `tracked_percent`. Dodano mapy nagrań dla zbiorów `Line`, `OnlyR` i `LineArc-1-2cm`.

## 7. Najważniejsze ograniczenie obecnego trackera kamery

Pomimo przechowywania pozycji landmarku jako trzech liczb, obecna mapa jest geometrycznie płaska. Funkcja `pixels_to_skin_plane()` prowadzi promień od środka kamery przez piksel, a następnie przecina go z płaszczyzną:

```text
Z = 0
```

To założenie pozwala wyznaczyć współrzędne punktu z jednej obserwacji. Bez znanej powierzchni pojedynczy piksel wyznacza tylko kierunek, a nie odległość punktu od kamery.

Na płaskim fantomie metoda może działać dobrze. Na palcu, ręce lub innej zakrzywionej powierzchni punkty zostaną umieszczone w błędnych miejscach. Samo usunięcie `Z=0` nie wystarczy — brakuje wtedy informacji o głębokości.

## 8. Rozważane sposoby przejścia do prawdziwej mapy 3D

### 8.1. Znany model powierzchni

Jeżeli znane jest równanie lub siatka powierzchni, promień wychodzący z piksela można przeciąć nie z `Z=0`, lecz z tym modelem. Przykładowo dla walca należy znaleźć punkt przecięcia promienia z powierzchnią walca.

Zaleta: jedna obserwacja może wystarczyć.

Wada: model musi być dobrze ustawiony w globalnym układzie i odpowiadać rzeczywistemu kształtowi skóry. Zmiany anatomii oraz nacisk sondy powodują błędy.

### 8.2. Wstępne zbudowanie nieruchomej mapy 3D

Najbardziej interesujący obecnie pomysł to osobne nagranie przygotowujące mapę:

1. Przed właściwym trackingiem kamera przejeżdża po całym obszarze.
2. Pozycja kamery w wybranych klatkach jest wyznaczana wyłącznie z ArUco.
3. Punkty skóry nie służą wtedy do lokalizowania kamery.
4. Te same punkty skóry są dopasowywane pomiędzy klatkami o znanych pozach.
5. Promienie z co najmniej dwóch różnych pozycji są triangulowane.
6. Kilka obserwacji jednego punktu pozwala poprawić jego pełne `X, Y, Z`.
7. Na końcu można wykonać bundle adjustment pozy kamer i landmarków.
8. Z kandydatów wybierany jest stabilny, równomiernie rozmieszczony zbiór.
9. Podczas właściwego badania mapa pozostaje zablokowana i służy tylko do dopasowania 3D–2D oraz PnP.

Matematycznie wystarczą dwie obserwacje punktu, ale w praktyce lepiej wymagać 3–5 dobrych obserwacji, odpowiedniej paralaksy i małego błędu reprojekcji. Sam obrót kamery nie wystarcza do dokładnego ustalenia głębokości — potrzebna jest translacja.

Ten pipeline może działać offline. Dzięki temu mapa nie musi jednocześnie lokalizować kamery i sama się budować. ArUco dostarcza niezależne pozy kamery, więc ogranicza dryft podczas mapowania.

### 8.3. Wymagania wobec markerów

Jeżeli pozycja kamery ma być liczona tylko z ArUco, każda użyta do triangulacji klatka musi widzieć przynajmniej jeden marker należący do wcześniej znanej wspólnej mapy markerów.

Wiele markerów może pojawiać się i znikać, ale ich pozycje muszą być znane w jednym układzie. Relację można uzyskać przez:

- wykonanie sztywnej ramy o geometrii znanej z CAD,
- wcześniejszą kalibrację, podczas której jednocześnie widać nakładające się zestawy markerów,
- ręczny dokładny pomiar ich pozycji.

Jeżeli marker A znika, tracking zostaje przerwany, a później pojawia się marker B o nieznanej pozycji względem A, powstaną dwie niezależne mapy.

## 9. Propozycja fizycznej ramy ArUco

Rozważana była duża kartka ze znacznikami i wyciętymi oknami odsłaniającymi skórę. Jest to dobry tani prototyp na płaskiej powierzchni, ale papier może się wyginać, marszczyć i naciskać na skórę. Po wygięciu zapisane relacje 3D markerów przestają być poprawne.

Lepszym rozwiązaniem jest sztywna rama lub dwie listwy umieszczone po bokach trasy sondy:

```text
markery  |        otwarty pas skóry        |  markery
markery  |       obszar ruchu sondy        |  markery
```

Zalecenia:

- pozycje markerów wynikają z projektu CAD,
- centralny obszar skóry pozostaje odsłonięty,
- markery mają różne identyfikatory,
- w klatce najlepiej widzieć 2–3 szeroko rozmieszczone markery,
- rama nie powinna dotykać ani deformować skóry,
- dla zakrzywionej powierzchni potrzebna jest sztywna przestrzenna konstrukcja, a nie dowolnie wygięta kartka.

Jeżeli rama nie przeszkadza sondzie, najlepiej pozostawić ją również podczas właściwego trackingu. ArUco może wtedy dostarczać absolutną pozę lub awaryjną relokalizację, a mapa skóry pozostaje źródłem dodatkowym.

## 10. Problem żelu USG i deformacji skóry

Statyczna mapa skóry ma poważne ograniczenie praktyczne. Żel oraz nacisk sondy mogą powodować:

- refleksy i nasycone obszary,
- smugi oraz pęcherzyki wykrywane jako cechy,
- zmianę wyglądu prawdziwych punktów skóry,
- załamanie światła i pozorne przesunięcie pikseli,
- fizyczne przesunięcie landmarków przez deformację skóry.

Zbudowanie mapy po nałożeniu żelu zmniejsza różnicę wyglądu początkowego, ale nie rozwiązuje problemu, ponieważ sonda podczas ruchu rozprowadza żel i odkształca skórę.

Rozważane rozwiązania:

- pozostawienie sztywnej ramy ArUco poza trasą sondy,
- kamera patrząca na obszar przed miejscem kontaktu sondy,
- odrzucanie refleksów i punktów poruszających się niespójnie,
- stałe rozproszone oświetlenie oraz zablokowane exposure i autofocus,
- polaryzacja krzyżowa ograniczająca refleksy,
- zewnętrzna kamera śledząca marker na sondzie,
- tracking elektromagnetyczny albo pozycja robota jako źródło absolutne.

Jeżeli skóra zmienia kształt, klasyczne sztywne PnP i nieruchoma mapa 3D przestają dokładnie opisywać scenę. Rozwiązaniem programowym byłby znacznie trudniejszy model deformowalny, dlatego praktycznie lepiej zapewnić zewnętrzne stabilne odniesienie.

## 11. Zbiory danych i wyniki

Aktualna struktura danych znajduje się pod `Data/<nazwa_zbioru>/` i obejmuje m.in.:

- `Line` — ruch liniowy,
- `OnlyR` — przede wszystkim zmiana orientacji,
- `LineWithR` — ruch liniowy ze zmianą orientacji,
- `LineArc-1-2cm` — nowe nagrania ruchu po łuku,
- starsze dane zachowane w osobnych katalogach.

Wyniki zwykłych uruchomień trafiają do `Camera/results/<DATA_FOLDER>/<nagranie>`, a wyniki batcha do `Camera/results_DISK_batch/<DATA_FOLDER>/<eksperyment>`.

Przy analizie wyników trzeba zawsze odróżniać:

- błąd samego trackingu,
- błędne przypisanie osi,
- różnicę między środkiem kamery i TCP Dobota,
- błędną orientację układów,
- brak synchronizacji,
- klatki `LOST`, dla których nie wolno interpolacją udawać poprawnego śledzenia.

## 12. Najważniejsze pliki

- [Camera/camera_tracking.py](Camera/camera_tracking.py) — główny tracker DISK + LightGlue.
- [Camera/skin_map_tracker.py](Camera/skin_map_tracker.py) — globalna mapa, PnP, rozszerzanie i pruning.
- [Camera/camera_tracking_hybrid.py](Camera/camera_tracking_hybrid.py) — uruchomienie wersji hybrydowej.
- [Camera/hybrid_skin_map_tracker.py](Camera/hybrid_skin_map_tracker.py) — przełączanie LightGlue/optical flow.
- [Camera/camera_tracking_optical_flow.py](Camera/camera_tracking_optical_flow.py) — osobna wersja optical flow.
- [Camera/tracking_visualization.py](Camera/tracking_visualization.py) — wykresy, filmy diagnostyczne i mapa z góry.
- [Camera/recording_axes.py](Camera/recording_axes.py) — mapowanie osi dla poszczególnych nagrań.
- [Camera/run_camera_tracking_batch.py](Camera/run_camera_tracking_batch.py) — eksperymenty seryjne i zbiorcze metryki.
- [IMU/calibrate_imu.py](IMU/calibrate_imu.py) — kalibracja drugiego IMU.
- [IMU/eskf.py](IMU/eskf.py) — ESKF i opcjonalna korekcja kamerą.
- [synchronize.py](synchronize.py) — synchronizacja czasów urządzeń.
- [generate_charuco_board.py](generate_charuco_board.py) — generowanie planszy ChArUco do kalibracji kamery.

## 13. Rekomendowana dalsza kolejność pracy

1. Zdecydować, czy docelowy system może pozostawić ramę ArUco podczas badania.
2. Wykonać sztywny prototyp ramy z otwartym pasem dla sondy.
3. Sprawdzić dokładność pozy kamery z jednego, dwóch i trzech widocznych markerów.
4. Zaimplementować osobny offline'owy skrypt budowy mapy 3D:
   - pozy kamery z ArUco,
   - matching punktów skóry między klatkami,
   - triangulacja,
   - odrzucanie słabej geometrii,
   - bundle adjustment,
   - równomierny wybór landmarków,
   - zapis mapy na dysk.
5. Zmienić właściwy tracker w tryb lokalizacji względem niezmiennej mapy 3D.
6. Osobno przetestować wpływ żelu na liczbę poprawnych dopasowań przy nieruchomej kamerze.
7. Wykonać fizyczną kalibrację transformacji kamera–IMU–punkt sondy.
8. Dopiero wtedy ponownie stroić fuzję ESKF, ponieważ bez wspólnych układów współrzędnych i wiarygodnych pomiarów strojenie kowariancji nie rozwiąże problemu.

## 14. Krótkie podsumowanie decyzji

- IMU bez korekcji nie zapewni stabilnej pozycji sondy.
- Globalna mapa jest lepsza od ciągłego sumowania optical flow, ponieważ pozwala wracać do wcześniejszych punktów.
- Obecny tracker działa na płaskiej mapie `Z=0`.
- Prawdziwe 3D wymaga modelu powierzchni albo wieloklatkowej triangulacji.
- Najczystszy sposób budowy mapy 3D to pozycje kamery z ArUco i triangulacja punktów skóry offline.
- Wszystkie markery muszą należeć do jednego znanego układu.
- Sztywna rama markerowa jest pewniejsza od wygiętej kartki.
- Żel i deformacja skóry mogą unieważnić statyczną mapę, dlatego potrzebne jest stabilne zewnętrzne odniesienie albo obserwacja obszaru poza miejscem kontaktu.
- Ostateczna poza musi zostać przeliczona na fizyczny punkt sondy, a nie pozostać pozą środka kamery.
