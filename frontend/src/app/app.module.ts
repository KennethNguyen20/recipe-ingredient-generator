import { NgModule } from '@angular/core';
import { BrowserModule } from '@angular/platform-browser';
import { RouterModule, Routes } from '@angular/router';
import { AppComponent } from './app.component';
import { RecipeComponent } from './recipe/recipe.component';

// Define routes
const routes: Routes = [
  { path: '', redirectTo: '/recipes', pathMatch: 'full' }, // Default route
  { path: 'recipes', component: RecipeComponent },
];

@NgModule({
  declarations: [
    AppComponent,
    RecipeComponent, // Declare the RecipeComponent here
  ],
  imports: [
    BrowserModule,
    RouterModule.forRoot(routes), // Import RouterModule with routes
  ],
  providers: [],
  bootstrap: [AppComponent], // Bootstrap with AppComponent
})
export class AppModule {}
